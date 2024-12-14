#  Copyright 2024 Microsoft. All rights reserved.
#  Licensed under the MSRLA License. See LICENSE in the repo root for license information.


import re
from typing import Any, TypeAlias

import numpy as np
from PIL import Image
from transformers import BaseImageProcessor, LlavaProcessor, PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature

SingleChatMessageType: TypeAlias = dict[str, str | int | None]
ChatMessageListType: TypeAlias = list[dict[str, str | list[SingleChatMessageType]]]
BoxType: TypeAlias = tuple[float, float, float, float]


class Maira2Processor(LlavaProcessor):
    """
    Constructs a Maira2 processor similar to LlavaProcessor but with additional arguments and functions to support
    multi-image grounded and non-grounded radiology report generation.

    In addition to the arguments of LlavaProcessor, Maira2Processor has the following extra arguments:

    Args:
        phrase_start_token (`str`, *optional*, defaults to `"<obj>"`):
            Special token used to denote the start of a grounded phrase (with or without box).
        phrase_end_token (`str`, *optional*, defaults to `"</obj>"`):
            Special token used to denote the end of a grounded phrase.
        box_start_token (`str`, *optional*, defaults to `"<box>"`):
            Special token used to denote the start of a bounding box.
        box_end_token (`str`, *optional*, defaults to `"</box>"`):
            Special token used to denote the end of a bounding box.
        num_box_coord_bins (`int`, *optional*, defaults to `100`):
            Number of bins used to represent the bounding box coordinates.
    """

    valid_kwargs = [
        "chat_template",
        "patch_size",
        "vision_feature_select_strategy",
        "image_token",
        "phrase_start_token",
        "phrase_end_token",
        "box_start_token",
        "box_end_token",
        "num_box_coord_bins",
    ]

    def __init__(
        self,
        image_processor: BaseImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        patch_size: int | None = None,
        vision_feature_select_strategy: str | None = None,
        chat_template: str | None = None,
        image_token: str = "<image>",
        phrase_start_token: str = "<obj>",
        phrase_end_token: str = "</obj>",
        box_start_token: str = "<box>",
        box_end_token: str = "</box>",
        num_box_coord_bins: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            **kwargs,
        )

        self.phrase_start_token = phrase_start_token
        self.phrase_end_token = phrase_end_token
        self.box_start_token = box_start_token
        self.box_end_token = box_end_token
        self.num_box_coord_bins = num_box_coord_bins

    @staticmethod
    def _normalize_image(image: Image.Image) -> Image.Image:
        """
        This function normalizes the input image to have pixel values in the range [0, 255].

        Args:
            image (Image.Image | np.ndarray):
                The input image to be normalized.

        Returns:
            Image.Image: The normalized image in grayscale.
        """
        image_np = np.array(image.convert("L"))
        image_np = image_np.astype(float)
        image_np -= image_np.min()
        image_np /= image_np.max()
        image_np *= 255
        image_np = image_np.astype(np.uint8)

        return Image.fromarray(image_np).convert("L")

    def _normalize_and_stack_images(
        self,
        current_frontal: Image.Image,
        current_lateral: Image.Image | None,
        prior_frontal: Image.Image | None,
    ) -> list[Image.Image]:
        """
        This function normalizes the input images and stacks them together. The images are stacked in the order of
        current_frontal, current_lateral, and prior_frontal. The order of images is important, since it must match the
        order of the images in the prompt, which is frontal, then lateral then prior.

        Args:
            current_frontal (Image.Image):
                The current frontal image.
            current_lateral (Image.Image | None):
                The current lateral image.
            prior_frontal (Image.Image | None):
                The prior frontal image.

        Returns:
            list[Image.Image]: The normalized images stacked together.
        """
        images = [self._normalize_image(current_frontal)]
        if current_lateral is not None:
            images.append(self._normalize_image(current_lateral))
        if prior_frontal is not None:
            images.append(self._normalize_image(prior_frontal))
        return images

    @staticmethod
    def _get_section_text_or_missing_text(section: str | None) -> str:
        """
        This function returns the input section text if it is not None and not empty, otherwise it returns a missing
        section text "N/A".

        Args:
            section (str | None):
                The input section text.

        Returns:
            str: The section text if it is not None and not empty, otherwise "N/A".
        """
        missing_section_text = "N/A"
        if not isinstance(section, str) or len(section) == 0:
            return missing_section_text
        return section

    @staticmethod
    def _construct_image_chat_messages_for_reporting(has_prior: bool, has_lateral: bool) -> list[SingleChatMessageType]:
        """
        This function constructs user chat messages based on the presence of the prior and lateral images.

        Args:
            has_prior (bool):
                A boolean indicating whether the prior image is present.
            has_lateral (bool):
                A boolean indicating whether the lateral image is present.

        Returns:
            list[SingleChatMessageType]: The image prompt messages in the form of a list of dictionaries.

        Example:

        ```python
        >>> _construct_image_chat_messages_for_reporting(has_prior=True, has_lateral=True)
        >>> # [
        >>> #     {"index": None, "text": "Given the current frontal image", "type": "text"},
        >>> #     {"index": 0, "text": None, "type": "image"},
        >>> #     {"index": None, "text": " the current lateral image", "type": "text"},
        >>> #     {"index": 1, "text": None, "type": "image"},
        >>> #     {"index": None, "text": " and the prior frontal image", "type": "text"},
        >>> #     {"index": 2, "text": None, "type": "image"},
        >>> # ]
        ```
        """

        def _add_single_image_to_chat_messages(prompt_text: str, image_index: int) -> None:
            image_prompt.extend(
                [
                    {"index": None, "text": prompt_text, "type": "text"},
                    {"index": image_index, "text": None, "type": "image"},
                ]
            )

        image_prompt: list[SingleChatMessageType] = []
        image_index = 0
        if not has_prior and not has_lateral:
            _add_single_image_to_chat_messages("Given the current frontal image only", image_index)
        else:
            _add_single_image_to_chat_messages("Given the current frontal image", image_index)
            image_index += 1
            if has_prior:
                if has_lateral:
                    _add_single_image_to_chat_messages(" the current lateral image", image_index)
                    image_index += 1
                _add_single_image_to_chat_messages(" and the prior frontal image", image_index)
            else:
                if has_lateral:
                    _add_single_image_to_chat_messages(" and the current lateral image", image_index)
        return image_prompt

    def _construct_chat_messages_reporting(
        self,
        has_prior: bool,
        has_lateral: bool,
        indication: str | None,
        technique: str | None,
        comparison: str | None,
        prior_report: str | None,
        get_grounding: bool = False,
        assistant_text: str | None = None,
    ) -> ChatMessageListType:
        """
        This function constructs the chat messages for reporting used in the grounded and non-grounded reporting tasks.

        Args:
            has_prior (bool):
                A boolean indicating whether the prior image is present.
            has_lateral (bool):
                A boolean indicating whether the lateral image is present.
            indication (str | None):
                The indication section text.
            technique (str | None):
                The technique section text.
            comparison (str | None):
                The comparison section text.
            prior_report (str | None):
                The prior report section text.
            get_grounding (bool):
                A boolean indicating whether to get the grounding information.
            assistant_text (str | None):
                The assistant text (can be set to None for ordinary inference).

        Returns:
            ChatMessageListType: The chat messages for reporting in the form of a list of dictionaries.

        Example:

        ```python
        >>> _construct_chat_messages_reporting(
        >>>     has_prior=True,
        >>>     has_lateral=True,
        >>>     indication="indication text from report goes here",
        >>>     technique="technique text from report goes here",
        >>>     comparison="comparison text from report goes here",
        >>>     prior_report="prior reporting text goes here",
        >>>     get_grounding=False,
        >>>     assistant_text=None,
        >>> )
        >>> # [
        >>> #     {"index": None, "text": "Given the current frontal image", "type": "text"},
        >>> #     {"index": 0, "text": None, "type": "image"},
        >>> #     {"index": None, "text": " the current lateral image", "type": "text"},
        >>> #     {"index": 1, "text": None, "type": "image"},
        >>> #     {"index": None, "text": " and the prior frontal image", "type": "text"},
        >>> #     {"index": 2, "text": None, "type": "image"},
        >>> #     {"index": None, "text": " PRIOR_REPORT: prior reporting text goes here", "type": "text"},
        >>> #     {"index": None, "text": " Provide a structured description of the impressions in the radiology study in comparison to the "
        >>> #     "prior frontal image. INDICATION: indication text from report goes here TECHNIQUE: technique text from report "
        >>> #     "goes here COMPARISON: comparison text from report goes here", "type": "text"},
        >>> # ]
        ```
        For Findings generation: Provide a structured description of the findings in ...
        """
        indication = self._get_section_text_or_missing_text(indication)
        technique = self._get_section_text_or_missing_text(technique)
        comparison = self._get_section_text_or_missing_text(comparison)
        prior_report = self._get_section_text_or_missing_text(prior_report)

        prompt = self._construct_image_chat_messages_for_reporting(has_prior=has_prior, has_lateral=has_lateral)

        if has_prior:
            prompt.append({"index": None, "text": f" PRIOR_REPORT: {prior_report}", "type": "text"})

        if get_grounding:
            prompt.append(
                {
                    "index": None,
                    "text": " Provide a structured description of the findings in the radiology study in comparison to the "
                    "prior frontal image. Each finding should be described as a self-contained plain-text sentence."
                    " If the finding is groundable, locate the finding in the current frontal chest X-ray image, "
                    "with bounding boxes indicating all locations where it can be seen in the current frontal "
                    "image. Otherwise, generate just the ungrounded finding without bounding boxes. INDICATION: "
                    f"{indication} TECHNIQUE: {technique} COMPARISON: {comparison}",
                    "type": "text",
                }
            )
        else:
            prompt.append(
                {
                    "index": None,
                    "text": " Provide a structured description of the findings in the radiology study in comparison to the "
                    f"prior frontal image. INDICATION: {indication} TECHNIQUE: {technique} COMPARISON: "
                    f"{comparison}",
                    "type": "text",
                }
            )
        messages: ChatMessageListType = [{"content": prompt, "role": "user"}]
        if assistant_text is not None:
            messages.append({"content": [{"index": None, "text": assistant_text, "type": "text"}], "role": "assistant"})
        return messages

    def _construct_chat_messages_phrase_grounding(
        self, phrase: str, assistant_text: str | None = None
    ) -> ChatMessageListType:
        """
        This function constructs the chat messages for phrase grounding used in the phrase grounding task.

        Args:
            phrase (str):
                The phrase to be grounded.
            assistant_text (str | None):
                The assistant text (can be set to None for ordinary inference).

        Returns:
            ChatMessageListType: The chat messages for phrase grounding in the form of a list of dictionaries.
        """
        prompt: list[SingleChatMessageType] = [
            {"index": None, "text": "Given the current frontal image", "type": "text"},
            {"index": 0, "text": None, "type": "image"},
            {
                "index": None,
                "text": f" Repeat the following finding as a grounded phrase with bounding boxes indicating all "
                f"locations where it can be seen in the given chest X-ray image. Finding: {phrase}",
                "type": "text",
            },
        ]
        messages: ChatMessageListType = [{"content": prompt, "role": "user"}]
        if assistant_text is not None:
            messages.append({"content": [{"index": None, "text": assistant_text, "type": "text"}], "role": "assistant"})
        return messages

    def format_reporting_input(
        self,
        current_frontal: Image.Image,
        current_lateral: Image.Image | None,
        prior_frontal: Image.Image | None,
        indication: str | None,
        technique: str | None,
        comparison: str | None,
        prior_report: str | None,
        get_grounding: bool = False,
        assistant_text: str | None = None,
    ) -> tuple[str, list[Image.Image]]:
        """
        This function formats the reporting prompt for the grounded and non-grounded reporting tasks from the given
        input images and text sections. The images are normalized and stacked together in the right order.

        Args:
            current_frontal (Image.Image):
                The current frontal image.
            current_lateral (Image.Image | None):
                The current lateral image.
            prior_frontal (Image.Image | None):
                The prior frontal image.
            indication (str | None):
                The indication section text.
            technique (str | None):
                The technique section text.
            comparison (str | None):
                The comparison section text.
            prior_report (str | None):
                The prior report section text.
            get_grounding (bool):
                A boolean indicating whether to construct the prompt for grounded or non-grounded reporting.
            assistant_text (str | None): The assistant text (can be set to None for ordinary inference).

        Returns:
            tuple[str, list[Image.Image]]: The formatted prompt text and the normalized images stacked in the right order.
        """
        images = self._normalize_and_stack_images(
            current_frontal=current_frontal,
            current_lateral=current_lateral,
            prior_frontal=prior_frontal,
        )
        messages = self._construct_chat_messages_reporting(
            has_prior=prior_frontal is not None,
            has_lateral=current_lateral is not None,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            get_grounding=get_grounding,
            assistant_text=assistant_text,
        )
        add_generation_prompt = assistant_text is None
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
        return text, images

    def format_phrase_grounding_input(
        self,
        frontal_image: Image.Image,
        phrase: str,
        assistant_text: str | None = None,
    ) -> tuple[str, list[Image.Image]]:
        """
        This function formats the phrase grounding prompt for the phrase grounding task from the given input
        image and phrase.

        Args:
            frontal_image (Image.Image):
                The frontal image.
            phrase (str):
                The phrase to be grounded.
            assistant_text (str | None):
                The assistant text (can be set to None for ordinary inference).

        Returns:
            tuple[str, list[Image.Image]]: The formatted phrase grounding prompt text and the normalized image.
        """
        images = self._normalize_and_stack_images(
            current_frontal=frontal_image,
            current_lateral=None,
            prior_frontal=None,
        )
        messages = self._construct_chat_messages_phrase_grounding(phrase)
        add_generation_prompt = assistant_text is None
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
        return text, images

    def format_and_preprocess_reporting_input(
        self,
        current_frontal: Image.Image,
        current_lateral: Image.Image | None,
        prior_frontal: Image.Image | None,
        indication: str | None,
        technique: str | None,
        comparison: str | None,
        prior_report: str | None,
        get_grounding: bool = False,
        assistant_text: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """
        This function formats and then preprocesses the input for the grounded and non-grounded reporting tasks from
        the given input images and text sections and returns the batch feature for the model. It calls format_reporting_input
        internally to format the input prompt and stack the images together in the right order.

        Args:
            current_frontal (Image.Image):
                The current frontal image.
            current_lateral (Image.Image | None):
                The current lateral image.
            prior_frontal (Image.Image | None):
                The prior frontal image.
            indication (str | None):
                The indication section text.
            technique (str | None):
                The technique section text.
            comparison (str | None):
                The comparison section text.
            prior_report (str | None):
                The prior report section text.
            get_grounding (bool):
                A boolean indicating whether to preprocess the input for grounded or non-grounded reporting.
            assistant_text (str | None):
                The assistant text (can be set to None for ordinary inference).

        Returns:
            BatchFeature: The batch feature for the model, ready to be passed to the model.

        """
        text, images = self.format_reporting_input(
            current_frontal=current_frontal,
            current_lateral=current_lateral,
            prior_frontal=prior_frontal,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            get_grounding=get_grounding,
            assistant_text=assistant_text,
        )
        return self(text=text, images=images, **kwargs)

    def format_and_preprocess_phrase_grounding_input(
        self,
        frontal_image: Image.Image,
        phrase: str,
        assistant_text: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """
        This function formats and then processes the input for the phrase grounding task from the given input image and
        phrase and returns the batch feature for the model. It calls format_phrase_grounding_input internally to format
        the input prompt and normalize the image.

        Args:
            frontal_image (Image.Image):
                The frontal image.
            phrase (str):
                The phrase to be grounded.
            assistant_text (str | None):
                The assistant text (can be set to None for ordinary inference).

        Returns:
            BatchFeature: The batch feature for the model, ready to be passed to the model.
        """
        text, images = self.format_phrase_grounding_input(
            frontal_image=frontal_image,
            phrase=phrase,
            assistant_text=assistant_text,
        )
        return self(text=text, images=images, **kwargs)

    def _get_text_between_delimiters(self, text: str, begin_token: str, end_token: str) -> list[str]:
        """
        This function splits the input text into a list of substrings beased on the given begin and end tokens.

        Args:
            text (str):
                The input text to be split.
            begin_token (str):
                The begin token.
            end_token (str):
                The end token.

        Returns:
            list[str]: The list of substrings between the given begin and end tokens.

        Example:

        ```python
        >>> _get_text_between_delimiters("<obj>This is a grounded phrase</obj>. <obj>This is another grounded phrase</obj>.", "<obj>", "</obj>")
        >>> # ["grounded phrase", "This is another grounded phrase"]

        >>> _get_text_between_delimiters("<box><x10><y20><x30><y40></box><box><x50><y60><x70><y80></box>", "<box>", "</box>")
        >>> # ["<x10><y20><x30><y40>", "<x50><y60><x70><y80>"]
        ```
        """
        split_text = []
        while begin_token in text:
            assert text.startswith(begin_token)
            end_index = text.find(end_token)
            assert end_index != -1
            split_text.append(text[len(begin_token) : end_index])
            text = text[end_index + len(end_token) :]
        assert len(text) == 0
        return split_text

    def convert_output_to_plaintext_or_grounded_sequence(
        self, text: str
    ) -> str | list[tuple[str, list[BoxType] | None]]:
        """
        This function converts the input text to a grounded sequence by extracting the grounded phrases and bounding
        boxes from the text. If the text is plaintext without any grounded phrases, it returns the text as is.

        Args:
            text (str):
                The input text to be converted.

        Returns:
            str | list[tuple[str, list[BoxType] | None]]: The grounded sequence.

        Example:

        ```python
        >>> convert_output_to_plaintext_or_grounded_sequence("<obj>grounded phrase <box><x55><y45><x70><y56></box></obj><obj>ungrounded phrase</obj>")
        >>> # [
        >>> #     ("grounded phrase", [(0.55, 0.45, 0.70, 0.56)]),
        >>> #     ("ungrounded phrase", None),
        >>> # ]

        >>> convert_output_to_plaintext_or_grounded_sequence("plain text")
        >>> # "plain text"
        ```
        """
        text = text.strip()

        # Plain text
        if not any(
            [
                self.phrase_start_token in text,
                self.phrase_end_token in text,
                self.box_start_token in text,
                self.box_end_token in text,
            ]
        ):
            return text

        # One or more grounded phrases
        grounded_phrase_texts = self._get_text_between_delimiters(text, self.phrase_start_token, self.phrase_end_token)
        grounded_phrases: list[tuple[str, list[BoxType] | None]] = []
        for grounded_phrase_text in grounded_phrase_texts:
            if self.box_start_token in grounded_phrase_text or self.box_end_token in grounded_phrase_text:
                first_box_start_index = grounded_phrase_text.find(self.box_start_token)
                phrase_text = grounded_phrase_text[:first_box_start_index].strip()
                boxes_text = grounded_phrase_text[first_box_start_index:]
                boxes_text_list = self._get_text_between_delimiters(
                    boxes_text, self.box_start_token, self.box_end_token
                )
                boxes: list[BoxType] = []
                for box_text in boxes_text_list:
                    # extract from <x_><y_><x_><y_>
                    regex = r"<x(\d+?)><y(\d+?)><x(\d+?)><y(\d+?)>"
                    match = re.search(regex, box_text)
                    if match:
                        x_min, y_min, x_max, y_max = match.groups()
                        box: BoxType = tuple(  # type: ignore[assignment]
                            (int(coord) + 0.5) / self.num_box_coord_bins for coord in (x_min, y_min, x_max, y_max)
                        )
                        assert all(0 <= coord <= 1 for coord in box), f"Invalid box coordinates: {box}"
                        boxes.append(box)
                    else:
                        raise ValueError(f"Invalid box coordinates: {box_text} not matching regex {regex}")
                grounded_phrases.append((phrase_text, boxes))
            else:
                grounded_phrases.append((grounded_phrase_text.lstrip(), None))
        return grounded_phrases

    @staticmethod
    def adjust_box_for_original_image_size(box: BoxType, width: int, height: int) -> BoxType:
        """
        This function adjusts the bounding boxes from the MAIRA-2 model output to account for the image processor
        cropping the image to be square prior to the model forward pass. The box coordinates are adjusted to be
        relative to the original shape of the image assuming the image processor cropped the image based on the length
        of the shortest side.

        Args:
            box (BoxType):
                The box to be adjusted, normalised to (0, 1).
            width (int):
                Original width of the image, in pixels.
            height (int):
                Original height of the image, in pixels.

        Returns:
            BoxType: The box normalised relative to the original size of the image.
        """
        crop_width = crop_height = min(width, height)
        x_offset = (width - crop_width) // 2
        y_offset = (height - crop_height) // 2

        norm_x_min, norm_y_min, norm_x_max, norm_y_max = box

        abs_x_min = int(norm_x_min * crop_width + x_offset)
        abs_x_max = int(norm_x_max * crop_width + x_offset)
        abs_y_min = int(norm_y_min * crop_height + y_offset)
        abs_y_max = int(norm_y_max * crop_height + y_offset)

        adjusted_norm_x_min = abs_x_min / width
        adjusted_norm_x_max = abs_x_max / width
        adjusted_norm_y_min = abs_y_min / height
        adjusted_norm_y_max = abs_y_max / height

        return (adjusted_norm_x_min, adjusted_norm_y_min, adjusted_norm_x_max, adjusted_norm_y_max)
