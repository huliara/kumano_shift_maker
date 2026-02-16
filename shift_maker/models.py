from django.db import models
from django.contrib.auth.models import (
    BaseUserManager,
    AbstractBaseUser,
    PermissionsMixin,
)
from datetime import date, time
from django.utils import timezone
import datetime

# Create your models here.


class WorkContent(models.Model):
    contentname = models.CharField(max_length=25)
    workload = models.IntegerField(default=0)
    detail = models.TextField(verbose_name="テキスト", blank=True)

    def __str__(self):
        return self.contentname


class Slot(models.Model):
    workname = models.CharField(max_length=30, null=True, blank=True)
    day = models.DateField(default=timezone.now)
    start_time = models.TimeField(default=time(10, 0, 0))
    end_time = models.TimeField(default=time(12, 0, 0))
    days_from_start = models.PositiveIntegerField(default=0)  # 1日目は０
    required_number = models.PositiveIntegerField(default=1)
    content = models.ForeignKey(
        WorkContent, on_delete=models.SET_DEFAULT, default="無し"
    )
    is_decided = models.BooleanField(
        default=False
    )  # 既に募集締め切りが過ぎた枠かどうか
    for_template = models.BooleanField(default=False)

    def __str__(self):
        return self.workname


class Block(models.TextChoices):
    a1 = "a1", "A1"
    a2 = "a2", "A2"
    a3 = "a3", "A3"
    a4 = "a4", "A4"
    b12 = "b12", "B12"
    b3 = "b3", "B3"
    b4 = "b4", "B4"
    c12 = "c12", "C12"
    c34 = "c34", "C34"
    all = "all", "All"


class UserManager(BaseUserManager):
    """ユーザーマネージャー."""

    use_in_migrations = True

    def _create_user(self, account_name, password, **extra_fields):
        if not account_name:
            raise ValueError("Users must have an name")
        account_name = self.model.normalize_username(account_name)
        user = self.model(account_name=account_name, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, account_name, password, **extra_fields):  #
        extra_fields.setdefault("is_staff", False)
        extra_fields.setdefault("is_superuser", False)
        return self._create_user(account_name, password, **extra_fields)  #

    def create_superuser(self, account_name, password, **extra_fields):  #
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self._create_user(account_name, password, **extra_fields)  #


class User(AbstractBaseUser, PermissionsMixin):
    Block_name = models.CharField(max_length=3, choices=Block.choices, blank=True)
    room_number = models.IntegerField(default=100)
    account_name = models.CharField(
        max_length=40, verbose_name="アカウント名", unique=True
    )
    workload_sum = models.IntegerField(verbose_name="過去の仕事量", default=0)
    assigned_work = models.ManyToManyField(
        WorkContent, verbose_name="経験済みの仕事", blank=True
    )
    assigning_slot = models.ManyToManyField(Slot, blank=True, related_name="slot_users")
    password = models.CharField(max_length=128, verbose_name="password")
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    objects = UserManager()
    USERNAME_FIELD = "account_name"

    def __str__(self):
        return self.account_name


class Shift(models.Model):
    shift_name = models.CharField(max_length=40, verbose_name="シフト名")
    first_day = models.DateField(default=timezone.now)
    deadline = models.DateField(default=timezone.now() + datetime.timedelta(days=3))
    slot = models.ManyToManyField(Slot, blank=True)
    target = models.CharField(max_length=3, choices=Block.choices)
    creater = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    is_decided = models.BooleanField(default=False)

    def __str__(self):
        return self.shift_name


class ShiftTemplate(models.Model):
    shift_template_name = models.CharField(max_length=40, verbose_name="テンプレート名")
    slot_templates = models.ManyToManyField(
        Slot,
        blank=True,
    )

    def __str__(self):
        return self.shift_template_name
