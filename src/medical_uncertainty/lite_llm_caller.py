import litellm


class LitellmCaller:

    def complete(self, kwargs):
        litellm.completion(**kwargs)