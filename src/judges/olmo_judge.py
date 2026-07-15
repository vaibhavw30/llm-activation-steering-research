"""Local OLMo-3 chat judge (open weights, no API key).

Mirrors the Anthropic `ask()` contract in judge_results.py: `.chat(system, user, max_tokens)`
returns the model's text, from which `extract_json` pulls the verdict. Because OLMo-3-Instruct is
a general instruction-following chat model (not a narrow yes/no head), it drives BOTH the `steer`
and `interpret` prompts unchanged.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_OLMO = "allenai/Olmo-3-7B-Instruct"


class OlmoJudge:
    def __init__(self, model=DEFAULT_OLMO, device="mps"):
        self.device = device
        self.model_name = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16).to(device).eval()

    @staticmethod
    def _build_messages(system, user):
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def chat(self, system, user, max_tokens=200):
        # transformers 5.x returns a BatchEncoding (input_ids + attention_mask) here, so pass the
        # dict through to generate() and slice the prompt off using input_ids' length.
        inputs = self.tok.apply_chat_template(
            self._build_messages(system, user),
            add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        return self.tok.decode(out[0][prompt_len:], skip_special_tokens=True)
