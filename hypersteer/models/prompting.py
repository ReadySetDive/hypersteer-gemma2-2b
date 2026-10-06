import torch

from hypersteer.models.model import Model
from hypersteer.models.modules.registry import register_model
from hypersteer.utils.helpers import get_logger

# Initialize the logger
logger = get_logger(__name__)


# This is copied from our LLM Judge evaluator for concept relevance.
# The sentence field will be filled with the actual output.
UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE = """[System]
Please act as an impartial judge and evaluate whether the specified concept is \
presented in the sentence fragment provided below. Focus solely on whether the concept is \
clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is \
clearly incorporated. Point out references to the concept in the fragment, noting any \
minimal or awkward incorporations. After providing your explanation, rate the concept's \
relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, \
1 indicates the concept is somewhat present but minimally or awkwardly incorporated, \
and 2 indicates the concept is more fully and effectively incorporated, with stronger and \
more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""


@register_model("PromptSteering")
class PromptSteering(Model):
    input_field = "steered_input"

    def __str__(self):
        return "PromptSteering"

    def load(self, dump_dir=None, **kwargs):
        pass

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass  # since we only need to train once for both methods.

    def train(self, examples, **kwargs):
        pass  # since we only need to train once for both methods.

    @torch.no_grad()
    def predict_step(self, batch_examples, batch_idx, **kwargs):
        self.model.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

        input_strings = batch_examples[self.input_field].tolist()
        # tokenize input_strings
        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        generations = self.model.generate(
            **inputs,
            max_new_tokens=eval_output_length,
            do_sample=True,
            temperature=temperature,
        )

        # Decode and print only the generated text without prompt tokens
        input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
        generated_texts = [
            self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
            for generation, input_length in zip(generations, input_lengths)
        ]

        # Calculate perplexity for each sequence
        batch_input_ids = self.tokenizer(
            generated_texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(self.device)
        batch_attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).float()

        # Forward pass without labels to get logits
        outputs = self.model(
            input_ids=batch_input_ids, attention_mask=batch_attention_mask
        )

        logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
        target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1

        # Calculate loss for each token
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Reshape losses and mask
        token_losses = token_losses.view(batch_input_ids.size(0), -1)
        mask = batch_attention_mask[:, 1:].contiguous()

        # Calculate perplexity for each sequence
        seq_lengths = mask.sum(dim=1)
        seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
        seq_perplexities = torch.exp(seq_losses).tolist()

        return {
            "generations": generated_texts,
            "perplexities": seq_perplexities,
        }


@register_model("SimplePromptSteering")
class SimplePromptSteering(PromptSteering):
    input_field = "simple_steered_input"

    def __str__(self):
        return "SimplePromptSteering"


@register_model("PromptDetection")
class PromptDetection(Model):
    input_field = "output"
    concept_field = "output_concept"

    def __str__(self):
        return "PromptDetection"

    def load(self, dump_dir=None, **kwargs):
        pass

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass  # since we only need to train once for both methods.

    def train(self, examples, **kwargs):
        pass  # since we only need to train once for both methods.

    def _get_rating_from_completion(self, completion):
        try:
            # Check if "Rating:" is in the completion
            if "Rating:" in completion:
                # Extract the part after "Rating:"
                rating_text = completion.split("Rating:")[-1].strip()
                # Take only the first line in case there's additional text
                rating_text = rating_text.split("\n")[0].strip()
                # Remove any extra characters around the number
                rating_text = (
                    rating_text.replace("[", "")
                    .replace("]", "")
                    .strip('"')
                    .strip("'")
                    .strip("*")
                    .strip()
                )
                # Convert to float and return the rating
                rating = float(rating_text)

                # Ensure the rating is within the expected range
                if rating < 0 or rating > 2:
                    raise ValueError(f"Invalid rating value: {rating}")
                return rating
            else:
                # Log warning and return default if "Rating:" is missing
                logger.warning(f"Cannot find rating value: {completion}")
                return -1  # DEFAULT_RATING
        except (ValueError, IndexError) as e:
            # Catch parsing errors and log them
            logger.error(
                f"Error parsing rating from completion: {completion}. Error: {e}"
            )
            return -1  # DEFAULT_RATING
