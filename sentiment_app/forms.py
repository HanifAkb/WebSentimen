import json
import re

from django import forms
from django.conf import settings
from django.contrib.auth import password_validation
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from django.utils.safestring import mark_safe

from .models import PredictionHistory, ScrapeHistory, SentimentModelVersion
from .services.model_service import available_model_versions, models_dir_path


def _upload_limit_mb() -> int:
    max_size = int(getattr(settings, "SENTIMENT_UPLOAD_MAX_SIZE", 5 * 1024 * 1024))
    one_mb = 1024 * 1024
    return max(1, (max_size + one_mb - 1) // one_mb)


def _model_version_choices() -> list[tuple[str, str]]:
    version_choices = available_model_versions()
    if version_choices:
        return version_choices
    return [("", "Tidak ada model tersedia")]


class PredictForm(forms.Form):
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
    model_version = forms.ChoiceField(
        required=True,
        label="Versi Model",
        choices=(),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_version"].choices = _model_version_choices()
        self.fields["upload_file"].help_text = f"Format CSV/TXT. Maksimal ukuran file: {_upload_limit_mb()} MB."
        for name, field in self.fields.items():
            if name == "model_version":
                field.widget.attrs["class"] = "form-select model-version-select"
            else:
                field.widget.attrs["class"] = "form-control"

    def clean(self):
        cleaned_data = super().clean()
        upload_file = cleaned_data.get("upload_file")
        if upload_file is None:
            raise forms.ValidationError("Unggah file CSV/TXT.")
        return cleaned_data


class TwitterFetchForm(forms.Form):
    LANGUAGE_CHOICES = (
        ("in", "Bahasa Indonesia"),
    )

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
    model_version = forms.ChoiceField(
        required=True,
        label="Versi Model",
        choices=(),
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
    language = forms.ChoiceField(
        required=True,
        label="Bahasa",
        choices=LANGUAGE_CHOICES,
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
        self.fields["model_version"].choices = _model_version_choices()
        for field_name, field in self.fields.items():
            existing_class = (field.widget.attrs.get("class") or "").strip()
            input_class = "form-select" if field_name in {"language", "model_version"} else "form-control"
            classes = f"{existing_class} {input_class}".strip()
            field.widget.attrs["class"] = " ".join(dict.fromkeys(classes.split()))
        self.fields["model_version"].widget.attrs["class"] = (
            f'{self.fields["model_version"].widget.attrs.get("class", "")} model-version-select'
        ).strip()

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


class AdminModelUploadForm(forms.Form):
    version_name = forms.CharField(
        label="Versi Model",
        max_length=100,
        help_text="Contoh: Sentimen V2.0",
    )
    knn_model_file = forms.FileField(
        label="File Model KNN",
        help_text="Unggah file .joblib untuk model KNN.",
    )
    svm_model_file = forms.FileField(
        label="File Model SVM",
        help_text="Unggah file .joblib untuk model SVM.",
    )

    def __init__(self, *args, **kwargs):
        self.existing_version_name = str(kwargs.pop("existing_version_name", "") or "").strip()
        super().__init__(*args, **kwargs)
        self.fields["version_name"].widget.attrs["class"] = "form-control"
        self.fields["version_name"].widget.attrs["placeholder"] = "Contoh: Sentimen V2.0"
        self.fields["knn_model_file"].widget.attrs["class"] = "form-control"
        self.fields["svm_model_file"].widget.attrs["class"] = "form-control"

    def clean_version_name(self):
        version_name = str(self.cleaned_data.get("version_name") or "").strip()
        if not version_name:
            raise forms.ValidationError("Isi versi model.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", version_name):
            raise forms.ValidationError("Versi model hanya boleh berisi huruf, angka, spasi, titik, garis bawah, atau tanda hubung.")
        if any(char in version_name for char in ("/", "\\", ":", "*", "?", '"', "<", ">", "|")):
            raise forms.ValidationError("Versi model mengandung karakter yang tidak diizinkan.")
        if (
            version_name != self.existing_version_name
            and SentimentModelVersion.objects.filter(version_name=version_name).exists()
        ):
            raise forms.ValidationError("Versi model tersebut sudah ada.")
        if version_name != self.existing_version_name and (models_dir_path() / version_name).exists():
            raise forms.ValidationError("Versi model tersebut sudah ada.")
        return version_name

    def _clean_joblib_file(self, field_name: str, label: str):
        uploaded_file = self.cleaned_data.get(field_name)
        if uploaded_file is None:
            return uploaded_file
        file_name = str(getattr(uploaded_file, "name", "") or "").strip()
        if not file_name.lower().endswith(".joblib"):
            raise forms.ValidationError(f"{label} harus berupa file .joblib.")
        return uploaded_file

    def clean_knn_model_file(self):
        return self._clean_joblib_file("knn_model_file", "File model KNN")

    def clean_svm_model_file(self):
        return self._clean_joblib_file("svm_model_file", "File model SVM")


class AdminModelEditForm(AdminModelUploadForm):
    knn_model_file = forms.FileField(
        required=False,
        label="File Model KNN",
        help_text="Kosongkan jika file KNN tidak diubah. Jika diisi, wajib berupa .joblib.",
    )
    svm_model_file = forms.FileField(
        required=False,
        label="File Model SVM",
        help_text="Kosongkan jika file SVM tidak diubah. Jika diisi, wajib berupa .joblib.",
    )


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


USERNAME_HELP_TEXT = "Maksimal 150 karakter. Gunakan huruf, angka, dan tanpa spasi."
PASSWORD_CONFIRM_HELP_TEXT = "Ulangi password yang sama."
PASSWORD_NEW_CONFIRM_HELP_TEXT = "Ulangi password baru yang sama."
ACTIVE_HELP_TEXT = "Menentukan apakah pengguna dapat login ke dalam sistem."
ROLE_HELP_TEXT = "Pilih satu peran untuk user: Staff atau Administrator."
ROLE_STAFF = "staff"
ROLE_ADMINISTRATOR = "administrator"
ROLE_CHOICES = (
    (ROLE_STAFF, "Staff"),
    (ROLE_ADMINISTRATOR, "Administrator"),
)


class AdminCreateUserForm(UserCreationForm):
    username = forms.CharField(label="Username", help_text=USERNAME_HELP_TEXT)
    full_name = forms.CharField(label="Nama Lengkap")
    email = forms.EmailField(required=False, label="Email")
    is_active = forms.BooleanField(
        required=False,
        initial=True,
        label="Aktif",
        help_text=ACTIVE_HELP_TEXT,
    )
    role = forms.ChoiceField(label="Peran", choices=ROLE_CHOICES, help_text=ROLE_HELP_TEXT)

    class Meta:
        model = User
        fields = (
            "username",
            "full_name",
            "email",
            "is_active",
            "role",
            "password1",
            "password2",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        text_fields = (
            "username",
            "full_name",
            "email",
            "password1",
            "password2",
        )
        for field_name in text_fields:
            self.fields[field_name].widget.attrs["class"] = "form-control"
        self.fields["role"].widget.attrs["class"] = "form-select role-select"
        self.fields["username"].widget.attrs.update({"placeholder": "username_baru", "autocomplete": "off"})
        self.fields["email"].widget.attrs.update({"placeholder": "opsional@email.com", "autocomplete": "off"})
        self.fields["password1"].widget.attrs["autocomplete"] = "new-password"
        self.fields["password2"].widget.attrs["autocomplete"] = "new-password"
        self.fields["password2"].help_text = PASSWORD_CONFIRM_HELP_TEXT
        self.fields["is_active"].widget.attrs["class"] = "form-check-input"

    def save(self, commit: bool = True):
        user = super().save(commit=False)
        user.first_name = self.cleaned_data.get("full_name", "")
        user.last_name = ""
        user.email = self.cleaned_data.get("email", "")
        user.is_active = bool(self.cleaned_data.get("is_active"))
        role = self.cleaned_data.get("role")
        user.is_superuser = role == ROLE_ADMINISTRATOR
        user.is_staff = True
        if commit:
            user.save()
        return user


class AdminEditUserForm(forms.ModelForm):
    full_name = forms.CharField(label="Nama Lengkap")
    role = forms.ChoiceField(label="Peran", choices=ROLE_CHOICES, help_text=ROLE_HELP_TEXT)
    password1 = forms.CharField(
        required=False,
        label="Password baru",
        strip=False,
        help_text=mark_safe(
            "<ul>"
            "<li>Password minimal 8 karakter.</li>"
            "<li>Password tidak boleh hanya berisi angka.</li>"
            "</ul>"
        ),
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )
    password2 = forms.CharField(
        required=False,
        label="Konfirmasi password baru",
        strip=False,
        help_text=PASSWORD_NEW_CONFIRM_HELP_TEXT,
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )

    class Meta:
        model = User
        fields = (
            "username",
            "full_name",
            "email",
            "is_active",
            "role",
        )
        labels = {
            "username": "Username",
            "full_name": "Nama Lengkap",
            "email": "Email",
            "is_active": "Aktif",
            "role": "Peran",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["full_name"].initial = self.instance.get_full_name() if self.instance.pk else ""
        for field_name in ("username", "full_name", "email"):
            self.fields[field_name].widget.attrs["class"] = "form-control"
        self.fields["username"].help_text = USERNAME_HELP_TEXT
        self.fields["role"].initial = ROLE_ADMINISTRATOR if self.instance.is_superuser else ROLE_STAFF
        self.fields["role"].widget.attrs["class"] = "form-select role-select"
        self.fields["is_active"].widget.attrs["class"] = "form-check-input"
        self.fields["is_active"].help_text = ACTIVE_HELP_TEXT
        for field_name in ("password1", "password2"):
            self.fields[field_name].widget.attrs["class"] = "form-control"

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
        user.first_name = self.cleaned_data.get("full_name", "")
        user.last_name = ""
        user.is_active = bool(self.cleaned_data.get("is_active"))
        role = self.cleaned_data.get("role")
        user.is_superuser = role == ROLE_ADMINISTRATOR
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
            "source_name",
            "text_column",
            "sample_count",
            "columns",
            "rows",
        )
        labels = {
            "user": "User",
            "source_name": "Sumber",
            "text_column": "Kolom Teks",
            "sample_count": "Jumlah Data",
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["user"].queryset = User.objects.order_by("username")
        self.fields["user"].widget.attrs["class"] = "form-select"
        for field_name in ("source_name", "text_column", "sample_count"):
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
            "resume_interval_days",
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
            "resume_interval_days": "Resume Interval Days",
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
            "resume_interval_days",
        ):
            self.fields[field_name].widget.attrs["class"] = "form-control"
        self.fields["is_complete"].widget.attrs["class"] = "form-check-input"
        self.fields["rows"].widget.attrs["class"] = "form-control font-monospace admin-json-field"

        if self.instance and self.instance.pk:
            self.fields["rows"].initial = _json_initial(self.instance.rows, [])

    def clean_rows(self):
        return self.cleaned_data.get("rows") or []
