"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx

from rxconfig import config

from typing import List, Tuple
from .src import PredictorModel
from .src import settings

# loading everything
global predictor
n_gram_checkpoint_path = "assets\\ngram_checkpoint_76000.pkl"
word_completor_checkpoint_path = "assets\\word_completor_dictionary.pkl"
predictor = PredictorModel(
    n_gram_checkpoint_path=n_gram_checkpoint_path,
    word_completor_checkpoint_path=word_completor_checkpoint_path,
)


class State(rx.State):
    """The app state."""

    text: str = ""
    alpha: float = 0.5
    num_words: int = 5
    num_texts: int = 3
    predictions: List[Tuple[float, str]] = []

    def update_predictions(self, **args):
        """Event handler to update the predictions in the background."""
        self.predictions = predictor.predict(
            self.text,
            alpha=self.alpha,
            n_words=self.num_words,
            n_texts=self.num_texts,
        )
        print("Final predictions:", self.text, "->", self.predictions)

    def prediction_list(self) -> List[str]:
        """Computed var that returns the list of predictions."""
        return [
            f"Probability: {p:.2f}, Predicted text: {t}" for p, t in self.predictions
        ]

    def set_text(self, text):
        self.text = text
        self.update_predictions()

    def set_alpha(self, alpha):
        try:
            self.alpha = float(alpha)
            self.alpha = min(max(self.alpha, 0), 1)  # Limit alpha to [0, 1]
            self.update_predictions()
        except ValueError:
            pass  # Ignore invalid input for alpha

    def set_num_words(self, num_words):
        try:
            self.num_words = int(num_words)
            self.num_words = min(
                max(self.num_words, 0), 10
            )  # Limit num_words to [0, 10]
            self.update_predictions()
        except ValueError:
            pass  # Ignore invalid input for num_words

    def set_num_texts(self, num_texts):
        try:
            self.num_texts = int(num_texts)
            self.num_texts = min(
                max(self.num_texts, 0), 10
            )  # Limit num_texts to [0, 10]
            self.update_predictions()
        except ValueError:
            pass  # Ignore invalid input for num_texts


def get_item(item):
    return rx.list.item(
        rx.hstack(
            rx.text(
                item[0],
                # on_click
                height="1.5em",
                background_color="pink",
                border="1px solid purple",
            ),
            rx.text(f"{item[1]:.2f}", font_size="1.25em", border="1px solid pink"),
        ),
    )


def index() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("Text: "),
                rx.input(
                    value=State.text,
                    on_change=State.set_text,
                    placeholder="Enter text",
                    background_color="white",
                    border="1px solid grey",
                    max_length=10000,
                    width="100%",
                    height="auto",
                ),
                rx.inset(
                    rx.image(
                        src="https://a.d-cd.net/TMYH_7LCS11_ixNRvm2QyPQ2Wjw-1920.jpg",
                        width="100%",
                        height="auto",
                    ),
                    side="top",
                    pb="current",
                    width="50%",
                ),
                width="100%",
            ),
            rx.hstack(
                rx.text("Alpha (0-1): "),
                rx.input(
                    value=State.alpha,
                    type_="number",
                    placeholder="Alpha",
                    on_change=State.set_alpha,
                    min=0,
                    max=1,
                    step=0.1,
                ),
            ),
            rx.hstack(
                rx.text("Num words (0-10): "),
                rx.input(
                    value=State.num_words,
                    type_="number",
                    placeholder="Num words",
                    on_change=State.set_num_words,
                    min=0,
                    max=10,
                    step=1,
                ),
            ),
            rx.hstack(
                rx.text("Num texts (0-10): "),
                rx.input(
                    value=State.num_texts,
                    type_="number",
                    placeholder="Num texts",
                    on_change=State.set_num_texts,
                    min=0,
                    max=10,
                    step=1,
                ),
            ),
            rx.divider(),
            rx.list.ordered(
                rx.foreach(
                    State.predictions,
                    get_item,
                ),
            ),
        ),
        bg="#ededed",
        padding="1em",
        border_radius="0.5em",
    )


app = rx.App()
app.add_page(index)
