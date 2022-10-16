# forms.py
from django import forms
from .models import OFAImage
  
class OFAImageForm(forms.ModelForm):
  
    class Meta:
        model = OFAImage
        fields = ['image']