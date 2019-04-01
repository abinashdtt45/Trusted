from django import forms

class NameForm(forms.Form):
    url = forms.CharField(label='Product URL :', max_length=1000)