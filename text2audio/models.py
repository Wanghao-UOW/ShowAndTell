from django.db import models  
class T2S(models.Model):
    image = models.ImageField(upload_to='') 