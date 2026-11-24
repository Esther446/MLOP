
from locust import HttpUser, task, between

class FlowerVisionUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_flower(self):
        with open("test_flower.jpg", "rb") as image:
            self.client.post(
                "/predict",
                files={"file": ("test_flower.jpg", image, "image/jpeg")},
                name="/predict"
            )
