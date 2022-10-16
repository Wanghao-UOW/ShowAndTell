# forms.py
from django import forms
from .models import T2S
  
class T2SForm(forms.ModelForm):
  
    class Meta:
        model = T2S
        fields = ['image']