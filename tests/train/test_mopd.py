import torch
import torch.nn.functional as F

from swift.rlhf_trainers.gkd_trainer import GKDTrainer, TeacherOutput, _mopd_tensor_from_response


def _make_mopd_trainer(weight, bias=None, *, beta=1.0, temperature=1.0, chunk_size=2):
    trainer = object.__new__(GKDTrainer)
    trainer._mopd_head_tensors = {
        'teacher_a': (weight.detach().cpu(), bias.detach().cpu() if bias is not None else None)
    }
    trainer.beta = beta
    trainer.temperature = temperature
    trainer.mopd_loss_chunk_size = chunk_size
    return trainer


def _teacher_target_kl(student_logits, teacher_logits, labels):
    mask = labels != -100
    student_log_probs = F.log_softmax(student_logits[mask], dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits[mask], dim=-1)
    return F.kl_div(student_log_probs, teacher_log_probs, reduction='none', log_target=True).sum() / mask.sum()


def test_mopd_mapping_parser():
    mapping = GKDTrainer._parse_mopd_mapping('math=http://m:8000,code=/tmp/code', required=True)
    assert mapping == {'math': 'http://m:8000', 'code': '/tmp/code'}


def test_mopd_teacher_weight_parser_defaults():
    weights = GKDTrainer._parse_mopd_teacher_weights(None)
    assert weights['reasoning'] == {'reasoning': 0.75, 'codegen': 0.05, 'agent': 0.20}
    assert weights['codegen'] == {'codegen': 0.75, 'reasoning': 0.05, 'agent': 0.20}
    assert weights['agent'] == {'agent': 0.75, 'reasoning': 0.05, 'codegen': 0.20}


def test_mopd_teacher_weight_parser_normalizes():
    weights = GKDTrainer._parse_mopd_teacher_weights(['codegen=codegen:3,reasoning:1'])
    assert weights == {'codegen': {'codegen': 0.75, 'reasoning': 0.25}}


def test_mopd_sample_weights_use_task_and_teacher_override():
    trainer = object.__new__(GKDTrainer)
    trainer.mopd_teacher_id_column = 'teacher_id'
    trainer.mopd_task_column = 'task'
    trainer.mopd_default_task = 'codegen'
    trainer.mopd_teacher_servers = {'reasoning': 'r', 'codegen': 'c', 'agent': 'a'}
    trainer.mopd_teacher_weights = GKDTrainer._parse_mopd_teacher_weights(None)

    weights = trainer._get_mopd_sample_teacher_weights([
        {
            'task': 'reasoning'
        },
        {
            'task': 'agent'
        },
        {
            'teacher_id': 'codegen',
            'task': 'reasoning'
        },
        {},
    ])

    assert weights[0] == {'reasoning': 0.75, 'codegen': 0.05, 'agent': 0.20}
    assert weights[1] == {'agent': 0.75, 'reasoning': 0.05, 'codegen': 0.20}
    assert weights[2] == {'codegen': 1.0}
    assert weights[3] == {'codegen': 0.75, 'reasoning': 0.05, 'agent': 0.20}


def test_mopd_load_lm_head_from_checkpoint(tmp_path):
    weight = torch.randn(7, 3)
    bias = torch.randn(7)
    ckpt = tmp_path / 'pytorch_model.bin'
    torch.save({'lm_head.weight': weight, 'lm_head.bias': bias}, ckpt)

    loaded_weight, loaded_bias = GKDTrainer._load_lm_head_tensors(str(tmp_path))
    assert torch.equal(loaded_weight, weight)
    assert torch.equal(loaded_bias, bias)


def test_mopd_response_tensor_parser():
    hidden = torch.randn(2, 4, 3)
    parsed_hidden, seq_lens = _mopd_tensor_from_response({'hidden_states': hidden, 'seq_lens': [4, 3]}, torch.float16)

    assert parsed_hidden.dtype == torch.float16
    assert parsed_hidden.shape == hidden.shape
    assert torch.equal(seq_lens, torch.tensor([4, 3]))


def test_mopd_loss_matches_direct_full_vocab_loss():
    torch.manual_seed(0)
    batch_size, seq_len, hidden_size, vocab_size = 2, 5, 4, 9
    weight = torch.randn(vocab_size, hidden_size)
    bias = torch.randn(vocab_size)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
    labels = torch.tensor([[-100, -100, 10, 11, 12], [-100, 20, 21, -100, -100]])

    trainer = _make_mopd_trainer(weight, bias)
    teacher_logits = torch.nn.functional.linear(teacher_hidden, weight, bias)
    expected = _teacher_target_kl(student_logits, teacher_logits, torch.roll(labels, shifts=-1, dims=1))
    actual = trainer._compute_mopd_loss(
        student_logits,
        TeacherOutput(hidden_states=teacher_hidden, teacher_ids=['teacher_a', 'teacher_a']),
        labels,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_mopd_loss_supports_teacher_specific_hidden_sizes():
    torch.manual_seed(1)
    seq_len, vocab_size = 4, 8
    weight_a = torch.randn(vocab_size, 3)
    weight_b = torch.randn(vocab_size, 5)
    student_logits = torch.randn(2, seq_len, vocab_size)
    hidden_a = torch.randn(1, seq_len, 3)
    hidden_b = torch.randn(1, seq_len, 5)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, -100, -100]])

    trainer = _make_mopd_trainer(weight_a)
    trainer._mopd_head_tensors['teacher_b'] = (weight_b, None)
    teacher_output = TeacherOutput(
        hidden_states={
            'teacher_a': {
                'indices': [0],
                'hidden_states': hidden_a
            },
            'teacher_b': {
                'indices': [1],
                'hidden_states': hidden_b
            },
        },
        teacher_ids=['teacher_a', 'teacher_b'],
    )

    actual = trainer._compute_mopd_loss(student_logits, teacher_output, labels)
    shifted_labels = torch.roll(labels, shifts=-1, dims=1)
    loss_a = _teacher_target_kl(
        student_logits[0:1], torch.nn.functional.linear(hidden_a, weight_a), shifted_labels[0:1])
    loss_b = _teacher_target_kl(
        student_logits[1:2], torch.nn.functional.linear(hidden_b, weight_b), shifted_labels[1:2])
    count_a = (shifted_labels[0:1] != -100).sum()
    count_b = (shifted_labels[1:2] != -100).sum()
    expected = (loss_a * count_a + loss_b * count_b) / (count_a + count_b)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_mopd_loss_matches_weighted_teacher_mixture():
    torch.manual_seed(2)
    batch_size, seq_len, hidden_size, vocab_size = 2, 4, 3, 7
    weight_a = torch.randn(vocab_size, hidden_size)
    weight_b = torch.randn(vocab_size, hidden_size)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    hidden_a = torch.randn(batch_size, seq_len, hidden_size)
    hidden_b = torch.randn(batch_size, seq_len, hidden_size)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, 4, 5, -100]])

    trainer = _make_mopd_trainer(weight_a)
    trainer._mopd_head_tensors['teacher_b'] = (weight_b, None)
    teacher_output = TeacherOutput(
        hidden_states={
            'teacher_a': {
                'indices': [0, 1],
                'hidden_states': hidden_a
            },
            'teacher_b': {
                'indices': [0, 1],
                'hidden_states': hidden_b
            },
        },
        teacher_weights=[{
            'teacher_a': 0.25,
            'teacher_b': 0.75
        }, {
            'teacher_a': 0.60,
            'teacher_b': 0.40
        }],
    )

    actual = trainer._compute_mopd_loss(student_logits, teacher_output, labels)
    shifted_labels = torch.roll(labels, shifts=-1, dims=1)
    mask = shifted_labels != -100
    student_log_probs = F.log_softmax(student_logits[mask], dim=-1)
    logits_a = F.linear(hidden_a, weight_a)
    logits_b = F.linear(hidden_b, weight_b)
    log_probs_a = F.log_softmax(logits_a[mask], dim=-1)
    log_probs_b = F.log_softmax(logits_b[mask], dim=-1)
    valid_sample_ids = mask.nonzero(as_tuple=False)[:, 0]
    weights_a = torch.tensor([teacher_output.teacher_weights[i.item()]['teacher_a'] for i in valid_sample_ids])
    weights_b = torch.tensor([teacher_output.teacher_weights[i.item()]['teacher_b'] for i in valid_sample_ids])
    mixed_teacher_log_probs = torch.logaddexp(
        log_probs_a + torch.log(weights_a)[:, None],
        log_probs_b + torch.log(weights_b)[:, None],
    )
    expected = F.kl_div(
        student_log_probs, mixed_teacher_log_probs, reduction='none', log_target=True).sum() / mask.sum()

    assert torch.allclose(actual, expected, atol=1e-6)
