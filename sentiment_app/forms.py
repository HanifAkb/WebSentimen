import json

from django import forms
from django.conf import settings
from django.contrib.auth import password_validation
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe

from .models import PredictionHistory, ScrapeHistory


def _upload_limit_mb() -> int:
    max_size = int(getattr(settings, "SENTIMENT_UPLOAD_MAX_SIZE", 5 * 1024 * 1024))
    one_mb = 1024 * 1024
    return max(1, (max_size + one_mb - 1) // one_mb)


class PredictForm(forms.Form):
    text_input = forms.CharField(
        required=False,
        label="Kalimat",
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


class ResumeScrapeForm(forms.Form):
    api_key = forms.CharField(
        label="API key twitterapi.io",
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Masukkan API key untuk melanjutkan scraping",
                "autocomplete": "off",
            },
            render_value=False,
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            existing_class = (field.widget.attrs.get("class") or "").strip()
            classes = f"{existing_class} form-control".strip()
            field.widget.attrs["class"] = " ".join(dict.fromkeys(classes.split()))


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
    first_name = forms.CharField(required=False, label="Nama depan")
    last_name = forms.CharField(required=False, label="Nama belakang")
    email = forms.EmailField(required=False, label="Email")
    is_active = forms.BooleanField(required=False, initial=True, label="Aktif")
    is_staff = forms.BooleanField(required=False, label="Staff")
    is_superuser = forms.BooleanField(required=False, label="Administrator")

    class Meta:
        model = User
        fields = (
            "username",
            "first_name",
            "last_name",
            "email",
            "is_active",
            "is_staff",
            "is_superuser",
            "password1",
            "password2",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        text_fields = (
            "username",
            "first_name",
            "last_name",
            "email",
            "password1",
            "password2",
        )
        for field_name in text_fields:
            self.fields[field_name].widget.attrs["class"] = "form-control"
        self.fields["username"].widget.attrs.update({"placeholder": "username_baru", "autocomplete": "off"})
        self.fields["email"].widget.attrs.update({"placeholder": "opsional@email.com", "autocomplete": "off"})
        self.fields["password1"].widget.attrs["autocomplete"] = "new-password"
        self.fields["password2"].widget.attrs["autocomplete"] = "new-password"
        for field_name in ("is_active", "is_staff", "is_superuser"):
            self.fields[field_name].widget.attrs["class"] = "form-check-input"

    def save(self, commit: bool = True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get("first_name", "")
        user.last_name = self.cleaned_data.get("last_name", "")
        user.email = self.cleaned_data.get("email", "")
        user.is_active = bool(self.cleaned_data.get("is_active"))
        user.is_superuser = bool(self.cleaned_data.get("is_superuser"))
        user.is_staff = bool(self.cleaned_data.get("is_staff")) or user.is_superuser
        if commit:
            user.save()
        return user


class AdminEditUserForm(forms.ModelForm):
    password1 = forms.CharField(
        required=False,
        label="Password baru",
        strip=False,
        help_text=mark_safe(
            "<ul>"
            "<li>Password minimal 8 karakter.</li>"
            "<li>Gunakan password yang tidak terlalu umum.</li>"
            "<li>Password tidak boleh hanya berisi angka.</li>"
            "</ul>"
        ),
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )
    password2 = forms.CharField(
        required=False,
        label="Konfirmasi password baru",
        strip=False,
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )

    class Meta:
        model = User
        fields = (
            "username",
            "first_name",
            "last_name",
            "email",
            "is_active",
            "is_staff",
            "is_superuser",
        )
        labels = {
            "username": "Username",
            "first_name": "Nama depan",
            "last_name": "Nama belakang",
            "email": "Email",
            "is_active": "Aktif",
            "is_staff": "Staff",
            "is_superuser": "Administrator",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in ("username", "first_name", "last_name", "email"):
            self.fields[field_name].widget.attrs["class"] = "form-control"
        for field_name in ("password1", "password2"):
            self.fields[field_name].widget.attrs["class"] = "form-control"
        for field_name in ("is_active", "is_staff", "is_superuser"):
            self.fields[field_name].widget.attrs["class"] = "form-check-input"

    def clean(self):
        cleaned_data = super().clean()
        password1 = cleaned_data.get("password1") or ""
        password2 = cleaned_data.get("password2") or ""
        if password1 or password2:
            if not password1 or not password2:
                raise forms.ValidationError("Isi password baru dan konfirmasi password baru.")
            if password1 != password2:
                self.add_error("password2", "Konfirmasi password tidak sama.")
            else:
                password_validation.validate_password(password1, self.instance)
        return cleaned_data

    def save(self, commit: bool = True):
        user = super().save(commit=False)
        if user.is_superuser:
            user.is_staff = True
        password = self.cleaned_data.get("password1")
        if password:
            user.set_password(password)
        if commit:
            user.save()
        return user


def _json_initial(value: object, fallback: object) -> str:
    if value in (None, ""):
        value = fallback
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


class AdminJSONField(forms.JSONField):
    def prepare_value(self, value):
        if isinstance(value, str):
            return value
        return _json_initial(value, [])


class AdminPredictionHistoryForm(forms.ModelForm):
    columns = AdminJSONField(
        required=False,
        label="Columns (JSON)",
        widget=forms.Textarea(attrs={"rows": 6}),
    )
    rows = AdminJSONField(
        required=False,
        label="Rows (JSON)",
        widget=forms.Textarea(attrs={"rows": 14}),
    )

    class Meta:
        model = PredictionHistory
        fields = (
            "user",
            "input_type",
            "text_input",
            "source_name",
            "text_column",
            "sample_count",
            "columns",
            "rows",
            "output_filename",
        )
        labels = {
            "user": "User",
            "input_type": "Tipe",
            "text_input": "Input Kalimat",
            "source_name": "Sumber",
            "text_column": "Kolom Teks",
            "sample_count": "Jumlah Data",
            "output_filename": "File Output",
        }
        widgets = {
            "text_input": forms.Textarea(attrs={"rows": 4}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["user"].queryset = User.objects.order_by("username")
        self.fields["user"].widget.attrs["class"] = "form-select"
        self.fields["input_type"].widget.attrs["class"] = "form-select"
        for field_name in ("text_input", "source_name", "text_column", "sample_count", "output_filename"):
            self.fields[field_name].widget.attrs["class"] = "form-control"
        for field_name in ("columns", "rows"):
            self.fields[field_name].widget.attrs["class"] = "form-control font-monospace admin-json-field"

        if self.instance and self.instance.pk:
            self.fields["columns"].initial = _json_initial(self.instance.columns, [])
            self.fields["rows"].initial = _json_initial(self.instance.rows, [])

    def clean_columns(self):
        return self.cleaned_data.get("columns") or []

    def clean_rows(self):
        return self.cleaned_data.get("rows") or []


class AdminScrapeHistoryForm(forms.ModelForm):
    rows = AdminJSONField(
        required=False,
        label="Rows (JSON)",
        widget=forms.Textarea(attrs={"rows": 14}),
    )

    class Meta:
        model = ScrapeHistory
        fields = (
            "user",
            "query",
            "language",
            "start_date",
            "end_date",
            "tweet_count",
            "rows",
            "is_complete",
            "resume_next_date",
            "stop_reason",
            "window_days",
        )
        labels = {
            "user": "User",
            "query": "Kueri",
            "language": "Bahasa",
            "start_date": "Tanggal Mulai",
            "end_date": "Tanggal Selesai",
            "tweet_count": "Jumlah Tweet",
            "is_complete": "Selesai",
            "resume_next_date": "Tanggal Lanjutan",
            "stop_reason": "Alasan Berhenti",
            "window_days": "Window Days",
        }
        widgets = {
            "query": forms.Textarea(attrs={"rows": 3}),
            "start_date": forms.DateInput(attrs={"type": "date"}),
            "end_date": forms.DateInput(attrs={"type": "date"}),
            "resume_next_date": forms.DateInput(attrs={"type": "date"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["user"].queryset = User.objects.order_by("username")
        self.fields["user"].widget.attrs["class"] = "form-select"
        for field_name in (
            "query",
            "language",
            "start_date",
            "end_date",
            "tweet_count",
            "resume_next_date",
            "stop_reason",
            "window_days",
        ):
            self.fields[field_name].widget.attrs["class"] = "form-control"
        self.fields["is_complete"].widget.attrs["class"] = "form-check-input"
        self.fields["rows"].widget.attrs["class"] = "form-control font-monospace admin-json-field"

        if self.instance and self.instance.pk:
            self.fields["rows"].initial = _json_initial(self.instance.rows, [])

    def clean_rows(self):
        return self.cleaned_data.get("rows") or []
