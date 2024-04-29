from django.db import models

# Create your models here.
class PredictedImage(models.Model):
    image = models.ImageField()
    name = models.CharField(max_length=100)
    predicted_label = models.CharField(max_length=100)

    def __str__(self):
        return self.name
