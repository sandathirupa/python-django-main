from django.db import models

# Create your models here.

class dataFields(models.Model):

    y_field             = models.CharField(max_length=200, blank=True)
    x_fields            = []
    modelType           = models.CharField(max_length=20, blank=True)
    modelName           = models.CharField(max_length=20, blank=True)
    objects             = models.Manager()

    def __str__(self):
        return self.y_field
