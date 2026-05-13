import argparse
import io
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoModel, AutoModelForCausalLM

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


class HiddenRequest(BaseModel):
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    position_ids: Optional[List[List[int]]] = None
    dtype: str = 'bf16'


def _dtype(name: str):
    return {
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'fp16': torch.float16,
        'float16': torch.float16,
        'fp32': torch.float32,
        'float32': torch.float32,
    }[name]


def _base_model(model):
    base_model = getattr(model, getattr(model, 'base_model_prefix', 'model'), None)
    return base_model if base_model is not None else model


def create_app(model_path: str, torch_dtype: str = 'bf16'):
    app = FastAPI()
    dtype = _dtype(torch_dtype)
    model_kwargs = {'torch_dtype': dtype, 'device_map': 'auto', 'trust_remote_code': True}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    except Exception:
        if AutoModelForImageTextToText is None:
            model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
        else:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs).eval()
    base_model = _base_model(model)
    device = next(model.parameters()).device

    @app.post('/v1/mopd/hidden_states')
    @torch.no_grad()
    def hidden_states(req: HiddenRequest):
        input_ids = torch.tensor(req.input_ids, dtype=torch.long, device=device)
        model_inputs = {'input_ids': input_ids, 'use_cache': False}
        if req.attention_mask is not None:
            model_inputs['attention_mask'] = torch.tensor(req.attention_mask, dtype=torch.long, device=device)
        if req.position_ids is not None:
            model_inputs['position_ids'] = torch.tensor(req.position_ids, dtype=torch.long, device=device)

        if base_model is model:
            model_inputs['output_hidden_states'] = True
        outputs = base_model(**model_inputs)
        hidden = getattr(outputs, 'last_hidden_state', None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        hidden = hidden.to(dtype=_dtype(req.dtype)).cpu().contiguous()
        if req.attention_mask is not None:
            seq_lens = torch.tensor([sum(mask) for mask in req.attention_mask], dtype=torch.long)
        else:
            seq_lens = torch.full((input_ids.shape[0], ), input_ids.shape[1], dtype=torch.long)

        buffer = io.BytesIO()
        torch.save({'hidden_states': hidden, 'seq_lens': seq_lens}, buffer)
        return Response(content=buffer.getvalue(), media_type='application/octet-stream')

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--torch_dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.torch_dtype), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
