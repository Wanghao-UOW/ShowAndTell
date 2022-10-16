from django.db import models  
class OFAImage(models.Model):
    image = models.ImageField(upload_to='') 