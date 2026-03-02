from django import forms
from django.conf import settings
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User


def _upload_limit_mb() -> int:
    max_size = int(getattr(settings, "SENTIMENT_UPLOAD_MAX_SIZE", 5 * 1024 * 1024))
    one_mb = 1024 * 1024
    return max(1, (max_size + one_mb - 1) // one_mb)


class PredictForm(forms.Form):
    text_input = forms.CharField(
        required=False,
        label="Kalimat tunggal",
        widget=forms.Textarea(
            attrs={
                "rows": 4,
                "placeholder": "Ketik satu kalimat/tweet untuk diklasifikasikan...",
            }
        ),
    )
    upload_file = forms.FileField(
        required=False,
        label="Unggah CSV/TXT",
    )
    text_column = forms.CharField(
        required=False,
        max_length=100,
        label="Kolom teks CSV (opsional)",
        help_text="Kosongkan untuk deteksi otomatis: text, tweet, content, sentence.",
        widget=forms.TextInput(attrs={"placeholder": "Contoh: text"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["upload_file"].help_text = f"Format CSV/TXT. Maksimal ukuran file: {_upload_limit_mb()} MB."
        for name, field in self.fields.items():
            if name == "upload_file":
                field.widget.attrs["class"] = "form-control"
            else:
                field.widget.attrs["class"] = "form-control"

    def clean(self):
        cleaned_data = super().clean()
        text_input = (cleaned_data.get("text_input") or "").strip()
        upload_file = cleaned_data.get("upload_file")
        if text_input and upload_file:
            raise forms.ValidationError("Gunakan salah satu: input teks tunggal ATAU unggah file, jangan keduanya.")
        if not text_input and not upload_file:
            raise forms.ValidationError("Isi satu kalimat atau unggah file CSV/TXT.")
        cleaned_data["text_input"] = text_input
        return cleaned_data


class TwitterFetchForm(forms.Form):
    api_key = forms.CharField(
        label="API key twitterapi.io",
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Masukkan API key Anda",
                "autocomplete": "off",
            },
            render_value=False,
        ),
    )
    query = forms.CharField(
        required=True,
        label="Kueri",
        help_text=(
            "Mendukung operator pencarian lanjutan."
        ),
        widget=forms.TextInput(
            attrs={
                "placeholder": "Masukkan kueri pencarian...",
            }
        ),
    )
    language = forms.CharField(
        required=False,
        max_length=10,
        label="Bahasa (opsional)",
        widget=forms.TextInput(attrs={"placeholder": "Contoh: 'in' atau 'en'"}),
    )
    start_date = forms.DateField(
        required=True,
        label="Tanggal mulai",
        widget=forms.DateInput(
            format="%Y-%m-%d",
            attrs={
                "type": "text",
                "placeholder": "dd/mm/yyyy",
                "class": "form-control js-flatpickr-date",
                "data-date-alt-format": "d/m/Y",
                "data-date-format": "Y-m-d",
                "autocomplete": "off",
            },
        ),
        input_formats=["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"],
        error_messages={"invalid": "Format tanggal harus dd/mm/yyyy."},
    )
    end_date = forms.DateField(
        required=True,
        label="Tanggal selesai",
        widget=forms.DateInput(
            format="%Y-%m-%d",
            attrs={
                "type": "text",
                "placeholder": "dd/mm/yyyy",
                "class": "form-control js-flatpickr-date",
                "data-date-alt-format": "d/m/Y",
                "data-date-format": "Y-m-d",
                "autocomplete": "off",
            },
        ),
        input_formats=["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"],
        error_messages={"invalid": "Format tanggal harus dd/mm/yyyy."},
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            existing_class = (field.widget.attrs.get("class") or "").strip()
            classes = f"{existing_class} form-control".strip()
            field.widget.attrs["class"] = " ".join(dict.fromkeys(classes.split()))

    def clean(self):
        cleaned_data = super().clean()
        query = (cleaned_data.get("query") or "").strip()
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")
        if not query:
            raise forms.ValidationError("Isi kueri.")
        if start_date and end_date and start_date > end_date:
            raise forms.ValidationError("Tanggal mulai tidak boleh lebih besar dari tanggal selesai.")
        cleaned_data["query"] = query
        return cleaned_data


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Username",
        widget=forms.TextInput(
            attrs={
                "autofocus": True,
                "autocomplete": "username",
                "class": "form-control",
                "placeholder": "Masukkan username",
            }
        ),
    )
    password = forms.CharField(
        label="Password",
        strip=False,
        widget=forms.PasswordInput(
            attrs={
                "autocomplete": "current-password",
                "class": "form-control",
                "placeholder": "Masukkan password",
            }
        ),
    )


class AdminCreateUserForm(UserCreationForm):
    email = forms.EmailField(required=False, label="Email")
    is_staff = forms.BooleanField(required=False, label="Akses admin Django")

    class Meta:
        model = User
        fields = ("username", "email", "is_staff", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(
            {
                "class": "form-control",
                "placeholder": "username_baru",
                "autocomplete": "off",
            }
        )
        self.fields["email"].widget.attrs.update(
            {
                "class": "form-control",
                "placeholder": "opsional@email.com",
                "autocomplete": "off",
            }
        )
        self.fields["password1"].widget.attrs.update(
            {
                "class": "form-control",
                "autocomplete": "new-password",
            }
        )
        self.fields["password2"].widget.attrs.update(
            {
                "class": "form-control",
                "autocomplete": "new-password",
            }
        )
        self.fields["is_staff"].widget.attrs.update(
            {
                "class": "form-check-input",
            }
        )

    def save(self, commit: bool = True):
        user = super().save(commit=False)
        user.email = self.cleaned_data.get("email", "")
        user.is_staff = bool(self.cleaned_data.get("is_staff"))
        if commit:
            user.save()
        return user
