import time

from ollama import Client


class ModelCall:
    def __init__(
        self,
        model_name,
        host="http://87.242.85.87:11434",
    ):
        self.client = Client(host=host)
        self.model_name = model_name

    def call(
        self,
        prompt,
        temperature=0.7,
        max_tokens=500,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
    ):
        # return {'response': 'Ah! The "dream neuro-fish" concept is a playful blend of **dreamlike storytelling** and **bio-tech analogies**—a perfect fusion for an imaginative narrative! Here’s how we can explore it:\n\n---\n\n### **Dream Neuro-Fruit Fly: A Sci-Fi Dream**\n\nIn this whimsical dream, you’re the **neuro-fruit fly**—an elegant, jelly-like creature with a special connection to your brain and digital world. As you navigate through your dream, you encounter challenges that test your ability to balance **neurochemical harmony** with the chaotic world of your "digital" environment.\n\n#### **Key Elements:**\n\n1. **Neurochemical Balance**:  \n   The neuro-fruit fly thrives in a delicate equilibrium between its biological instincts and the digital chaos it inhabits. For example, when you’re trying to *fix*', 'response_time_sec': 24.000548999999864, 'generation_time_sec': 21.050604, 'response_length_chars': 785, 'response_length_tokens': 500, 'prompt_tokens': 18, 'prompt_eval_time_sec': 0.121786, 'load_time_sec': 2.089587652}
        start_time = time.perf_counter()

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            stream=False,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
            },
        )

        end_time = time.perf_counter()

        answer = response.response

        return {
            "response": answer,

            "response_time_sec": end_time - start_time,

            "generation_time_sec": response.eval_duration / 1e9,

            "response_length_chars": len(answer),
            "response_length_tokens": response.eval_count,

            "prompt_tokens": response.prompt_eval_count,
            "prompt_eval_time_sec": response.prompt_eval_duration / 1e9,
            "load_time_sec": response.load_duration / 1e9,
        }
