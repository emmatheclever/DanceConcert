from manim import *

class ExplainerVid(Scene):
    def construct(self):
        # Intro
        question2 = Text("to merge the intuition of a dancer")
        question1 = Text("What would it look like").next_to(question2, UP)
        question3 = Text("with the organization of a computer?").next_to(question2, DOWN)
        self.add(question1, question2, question3)
        self.play(FadeOut(question1, question2, question3, scale=1.5))
        self.wait(2)

        # What's a Vector?
        plane = NumberPlane()

        vec_1 = Vector([1, 2])
        self.add(plane, vec_1)
        self.play(FadeIn(plane, vec_1))
        self.play(vector_to_coords(vec_1, integer_labels=True, clean_up=True))
