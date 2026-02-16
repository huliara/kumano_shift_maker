from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404, redirect
from django.urls.base import reverse_lazy
from django.views.generic import (
    CreateView,
    FormView,
    TemplateView,
    ListView,
    UpdateView,
)
from django.urls import reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import PasswordChangeView, PasswordChangeDoneView
from django.db.models import Count, F
from .models import Slot, Shift, User, ShiftTemplate, WorkContent
from .forms import ShiftForm, ShiftFormFromTemplate, MyPasswordChangeForm
from datetime import timedelta
import pandas as pd
import numpy as np
from ortoolpy.etc import addvar
from pulp import *
from ortoolpy import addvars, addbinvars
from django_pandas.io import read_frame
import datetime
from django.contrib import messages
from django.contrib.auth.decorators import login_required


# Create your views here.
# 回答フォームのビュー
@login_required
def shift_recruit_view(request, pk):
    shift = get_object_or_404(Shift, pk=pk)
    if shift.is_decided:
        return HttpResponseRedirect(
            reverse("shift_maker:result_schedule", args=[shift.pk])
        )
    forms = ShiftForm(request.POST or None, instance=shift)
    if forms.is_valid():
        forms.save()
    days_list = sorted(list(set(list(shift.slot.all().values_list("day", flat=True)))))
    time_list = sorted(
        list(set(list(shift.slot.all().values_list("start_time", "end_time"))))
    )
    sametime_slotlist = []
    for start_time, end_time in time_list:
        time_slot_list = []
        for day in days_list:
            slots = shift.slot.filter(day=day, start_time=start_time, end_time=end_time)
            time_slot_list.append(slots)
        sametime_slotlist.append((start_time, end_time, time_slot_list))
    return render(
        request,
        "shift_maker/answer.html",
        {
            "forms": forms,
            "shift": shift,
            "days_list": days_list,
            "sametime_slot_list": sametime_slotlist,
        },
    )


# 回答処理用の関数
def shift_receive_answer_view(request, pk):
    user = request.user
    answer = request.POST.getlist("slot")
    testslots = user.assigning_slot.all()
    for answer_slot_id in answer:
        slot = Slot.objects.get(id=answer_slot_id)
        user.assigning_slot.add(slot)
        user.save()
    for slot in testslots:
        print(slot.workname)
    return HttpResponseRedirect(reverse("shift_maker:mypage"))


def shift_recruit_detail(request, pk):
    shift = get_object_or_404(Shift, pk=pk)
    days_list = sorted(list(set(list(shift.slot.all().values_list("day", flat=True)))))
    time_list = sorted(
        list(set(list(shift.slot.all().values_list("start_time", "end_time"))))
    )
    sametime_slotlist = []
    for start_time, end_time in time_list:
        time_slot_list = []
        for day in days_list:
            slots = shift.slot.filter(day=day, start_time=start_time, end_time=end_time)
            one_table = []
            for slot in slots:
                one_table.append((slot, slot.slot_users.all()))
            time_slot_list.append(one_table)
        sametime_slotlist.append((start_time, end_time, time_slot_list))
    return render(
        request,
        "shift_maker/recruit_detail.html",
        {
            "shift": shift,
            "days_list": days_list,
            "sametime_slot_list": sametime_slotlist,
        },
    )


class CreateSlot(CreateView):
    model = Slot
    fields = "__all__"
    success_url = reverse_lazy("shift_maker:mypage")


class CreateShiftTemplate(CreateView):
    model = ShiftTemplate
    fields = "__all__"

    def form_valid(self, form):
        object = form.save(commit=False)
        object.user = self.request.user
        object.save()
        return super().form_valid(form)


class CreateShift(CreateView):
    model = Shift
    fields = ["shift_name", "first_day", "deadline", "slot", "target"]
    success_url = reverse_lazy("shift_maker:mypage")

    def form_valid(self, form):
        object = form.save(commit=False)
        object.creater = self.request.user
        object.save()
        Slot.objects.filter(day__lt=datetime.datetime.today()).delete()
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["form"].fields["slot"].queryset = Slot.objects.filter(
            for_template=False
        )
        return context


class UserUpdate(UpdateView):
    template_name = "shift_maker/member_update_form.html"
    model = User
    fields = ["Block_name", "room_number", "password"]

    def get_success_url(self):
        return reverse("shift_maker:mypage")

    def get_form(self):
        form = super(UserUpdate, self).get_form()
        form.fields["Block_name"].label = "ブロック名"
        form.fields["room_number"].label = "部屋番号"
        form.fields["password"].label = "パスワード"
        return form


class PasswordChange(PasswordChangeView):
    form_class = MyPasswordChangeForm
    success_url = reverse_lazy("shift_maker:password_change_done")
    template_name = "shift_maker/password_change.html"


class PasswordChangeDone(PasswordChangeDoneView):
    template_name = "shift_maker/password_change_done.html"


class ShiftFormFromTemplateView(FormView):
    template_name = "shift_maker/templateconvert.html"
    form_class = ShiftFormFromTemplate
    success_url = reverse_lazy("shift_maker:mypage")

    def form_valid(self, form):
        Slot.object.filter(day__lt=datetime.datetime.today()).delete()
        return super().form_valid(form)


# 計算しても加算されない
class MyPageView(LoginRequiredMixin, TemplateView):
    model = User
    template_name = "shift_maker/mypage.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["decided_assign_slot"] = self.request.user.assigning_slot.filter(
            is_decided=True
        )
        context["undecided_assign_slot"] = self.request.user.assigning_slot.filter(
            is_decided=False
        )
        context["lack_slot"] = Slot.objects.annotate(
            assigning_number=Count("slot_users")
        ).filter(assigning_number__lt=F("required_number"), is_decided=True)
        context["shifts"] = Shift.objects.filter(is_decided=False)
        context["decided_shifts"] = Shift.objects.filter(is_decided=True)
        context["made_shifts"] = Shift.objects.filter(
            creater=self.request.user, is_decided=False
        )
        context["others_shifts"] = Shift.objects.exclude(
            creater=self.request.user, is_decided=True
        )
        return context


class BlockMemberList(ListView):
    model = User

    def get_queryset(self):
        return (
            User.objects.filter(Block_name=self.request.user.Block_name)
            .order_by("-workload_sum")
            .all()
        )


class WorkContentList(ListView):
    model = WorkContent


# シフトのテンプレートからShiftModelを作る関数
# TODO 不正なフォームへの対応
def shift_from_template(request):
    form = ShiftFormFromTemplate(data=request.POST)
    if request.method == "POST":
        if form.is_valid():
            selected_shift_template = form.cleaned_data.get("shift_template")
            slots = selected_shift_template.slot_templates.all()
            first_day = form.cleaned_data.get("first_day")
            shift_name = form.cleaned_data.get("shift_name")
            deadline = form.cleaned_data.get("deadline")
            target = form.cleaned_data.get("target")
            creater = request.user
            shift = Shift.objects.create(
                shift_name=shift_name,
                first_day=first_day,
                deadline=deadline,
                target=target,
                creater=creater,
            )
            for slot in slots:
                slot.day = first_day + timedelta(days=slot.days_from_start)
                slot.id = None
                slot.save()
                slot.workname = slot.workname + "(" + str(slot.id) + ")"
                slot.is_decided = False
                slot.save()
                shift.slot.add(slot)
            messages.success(request, "作成成功", fail_silently=True)
            return HttpResponseRedirect(reverse("shift_maker:mypage"))
        elif request.method == "GET":
            messages.error(request, "不正なフォームです", fail_silently=True)
            return HttpResponseRedirect(
                reverse("shift_maker:shift_create_form_template")
            )
        else:
            messages.error(request, "不正なフォームです", fail_silently=True)
            return HttpResponseRedirect(
                reverse("shift_maker:shift_create_form_template")
            )
    else:
        messages.error(request, "不正なフォームです", fail_silently=True)
        return HttpResponseRedirect(reverse("shift_maker:shift_create_form_template"))


# 制約条件ごとのテストは完了
def shift_calculate(request, pk):
    shift = get_object_or_404(Shift, pk=pk)
    # モデルインスタンスのフィールドの値をvalues_list で取り出してそれをlist()でlist化して
    # Pandasの列又は行に追加している
    # values_list で値を取り出した際の順序がどのようになっているのかわかっていないので順序が
    # おかしくなっている可能性あり
    slots = shift.slot.all()
    slot_df = read_frame(
        slots,
        fieldnames=["required_number", "content__workload", "content__id"],
        index_col="id",
    )
    users = User.objects.filter(Block_name=shift.target)
    users_df = read_frame(users, fieldnames=["workload_sum"], index_col="id")
    user_ids = users.values_list("id", flat=True)
    user_list = list(user_ids)
    df = pd.DataFrame(index=slot_df.index, columns=user_list)
    df.fillna(0, inplace=True)
    workcontents = []
    for slot in slots:
        workcontents.append(slot.content.id)
        assigning_workers = slot.slot_users.all()
        for assigning_worker in assigning_workers:
            df.at[slot.id, assigning_worker.id] = 1
    list(set(workcontents))
    exp_df = pd.DataFrame(index=workcontents, columns=user_list)
    exp_df.fillna(0, inplace=True)
    # 経験済みの仕事を列挙している
    for user_id in user_list:
        usermodel = User.objects.get(id=user_id)
        assigned_works = usermodel.assigned_work.all()
        for assigned_work in assigned_works:
            index = assigned_work.id
            if index in workcontents:
                exp_df.at[index, user_id] = 1
    # 時間帯が重複しているシフト枠の組を取り出す処理。日付ごとに行っている
    overlapping_pairs = overlapping_slots(slots)
    var = pd.DataFrame(
        np.array(addbinvars(len(slot_df.index), len(user_list))),
        index=slot_df.index,
        columns=user_list,
    )
    shift_rev = df[df.columns].apply(lambda r: 1 - r[df.columns], 1)
    k = LpProblem()

    C_need_diff_over = 11
    C_need_diff_shortage = 1000
    C_experience = 10
    C_minmax = 10
    # 希望していない枠に入らないようにする制約条件
    for (_, h), (_, n) in zip(shift_rev.iterrows(), var.iterrows()):
        k += lpDot(h, n) <= 0
    # 同じ時間帯の枠に同じ人が入らないようにする制約条件
    # エラー発生中
    for index, r in var.items():
        print(r)
        print(overlapping_pairs)
        for i in range(len(overlapping_pairs)):
            print(r[overlapping_pairs[i][0]], r[overlapping_pairs[i][1]])
            k += r[overlapping_pairs[i][0]] + r[overlapping_pairs[i][1]] <= 1
    df["V_need_dif_over"] = addvars(len(slot_df.index))
    df["V_need_dif_shortage"] = addvars(len(slot_df.index))
    df["V_experience"] = addvars(len(slot_df.index))
    V_worksum_diff = addvar()
    # 必要な人数と実際に入る人数の差に対する制約条件
    for (_, r), (index, d) in zip(df.iterrows(), var.iterrows()):
        k += r.V_need_dif_over >= (lpSum(d) - slot_df.at[index, "required_number"])
        k += r.V_need_dif_shortage >= -(lpSum(d) - slot_df.at[index, "required_number"])
    # 経験者が少なくとも一人入る制約条件
    for (_, r), (index, d) in zip(df.iterrows(), var.iterrows()):
        k += (
            lpDot(exp_df.loc[slot_df.at[index, "content__id"]], d) + r.V_experience >= 1
        )
    # 合計仕事量が均等になるようにする制約条件
    for column, w in var.items():
        k += (
            lpDot(slot_df["content__workload"], w) + users_df.at[column, "workload_sum"]
            <= V_worksum_diff
        )
    # コストを計算
    k += (
        C_need_diff_over * lpSum(df.V_need_dif_over)
        + C_need_diff_shortage * lpSum(df.V_need_dif_shortage)
        + C_experience * lpSum(df.V_experience)
        + C_minmax * V_worksum_diff
    )
    k.solve()
    result_np = np.vectorize(value)(var).astype(int)
    result = pd.DataFrame(result_np, index=slot_df.index, columns=user_list)
    print(result)
    print(user_list)
    for column in user_list:
        user = User.objects.get(id=column)
        for index in result.index:
            if result.at[index, column] == 0:
                slot = Slot.objects.get(id=index)
                user.assigning_slot.remove(slot)
            else:
                slot = Slot.objects.get(id=index)
                workload = Slot.objects.values_list("content__workload", flat=True).get(
                    id=index
                )
                user.workload_sum += workload
                user.save(update_fields=["workload_sum"])
    for slot in slots:
        slot.is_decided = True
    Slot.objects.bulk_update(slots, ["is_decided"])
    shift.is_decided = True
    shift.save()
    messages.success(request, "シフト作成完了", fail_silently=True)
    return HttpResponseRedirect(reverse("shift_maker:result_schedule", args=[shift.pk]))


@login_required
def shift_calculate_result(request, pk):
    shift = get_object_or_404(Shift, pk=pk)
    days_list = sorted(list(set(list(shift.slot.all().values_list("day", flat=True)))))
    time_list = sorted(
        list(set(list(shift.slot.all().values_list("start_time", "end_time"))))
    )
    sametime_slotlist = []
    for start_time, end_time in time_list:
        time_slot_list = []
        for day in days_list:
            slots = shift.slot.filter(day=day, start_time=start_time, end_time=end_time)
            one_table = []
            for slot in slots:
                one_table.append((slot, slot.slot_users.all()))
            time_slot_list.append(one_table)
        sametime_slotlist.append((start_time, end_time, time_slot_list))
    return render(
        request,
        "shift_maker/shift_result.html",
        {
            "shift": shift,
            "days_list": days_list,
            "sametime_slot_list": sametime_slotlist,
        },
    )


def assign_content(request, pk):
    content = WorkContent.objects.get(pk=pk)
    user = request.user
    user.assigned_work.add(content)
    messages.success(request, "%s 登録完了" % content.contentname, fail_silently=True)
    return HttpResponseRedirect(reverse("shift_maker:contentlist"))


# 人数不足スロットの登録処理


# テスト１
def assign_lack_slot(request, pk):
    slot = Slot.objects.get(pk=pk)
    user = request.user
    if slot in user.assigning_slot.all():
        messages.warning(request, "既にこの仕事に入っています", fail_silently=True)
        return HttpResponseRedirect(reverse("shift_maker:mypage"))
    workload = Slot.objects.values_list("content__workload", flat=True).get(pk=pk)
    user.assigning_slot.add(slot)
    if overlapping_slots(user.assigning_slot.all()):
        user.assigning_slot.remove(slot)
        messages.warning(request, "同じ時間帯に仕事に入っています", fail_silently=True)
        return HttpResponseRedirect(reverse("shift_maker:mypage"))
    user.workload_sum += workload
    user.save()
    messages.success(request, "%s 登録しました" % slot.workname, fail_silently=True)
    return HttpResponseRedirect(reverse("shift_maker:mypage"))


def assign_slot(request, pk):
    slot = Slot.objects.get(pk=pk)
    user = request.user
    if slot in user.assigning_slot.all():
        messages.warning(request, "既にこの仕事に入っています", fail_silently=True)
        return redirect(request.META["HTTP_REFERER"])
    workload = Slot.objects.values_list("content__workload", flat=True).get(pk=pk)
    user.assigning_slot.add(slot)
    if overlapping_slots(user.assigning_slot.all()):
        user.assigning_slot.remove(slot)
        messages.warning(request, "同じ時間帯に仕事に入っています", fail_silently=True)
        return redirect(request.META["HTTP_REFERER"])
    user.workload_sum += workload
    user.save()
    messages.success(request, "%s 登録しました" % slot.workname, fail_silently=True)
    return redirect(request.META["HTTP_REFERER"])


def replace_slot(request, slot_id, user_id):
    slot = Slot.objects.get(pk=slot_id)
    user = request.user
    target = User.objects.get(pk=user_id)
    if slot in user.assigning_slot.all():
        messages.warning(request, "既にこの仕事に入っています", fail_silently=True)
        return redirect(request.META["HTTP_REFERER"])
    workload = Slot.objects.values_list("content__workload", flat=True).get(pk=slot_id)
    user.assigning_slot.add(slot)
    if overlapping_slots(user.assigning_slot.all()):
        user.assigning_slot.remove(slot)
        messages.warning(request, "同じ時間帯に仕事に入っています", fail_silently=True)
        return redirect(request.META["HTTP_REFERER"])
    user.workload_sum += workload
    user.save()
    target.assigning_slot.remove(slot)
    target.workload_sum -= workload
    target.save()
    messages.success(request, "%s 交代しました" % slot.workname, fail_silently=True)
    return redirect(request.META["HTTP_REFERER"])


# 登録済みスロットの登録解除
def delete_assigned_slot(request, pk):
    slot = Slot.objects.get(pk=pk)
    user = request.user
    workload = Slot.objects.values_list("content__workload", flat=True).get(pk=pk)
    user.assigning_slot.remove(slot)
    user.workload_sum -= workload
    user.save()
    messages.success(request, "%s 登録解除しました" % slot.workname, fail_silently=True)
    return HttpResponseRedirect(reverse("shift_maker:mypage"))


# 予約済みスロットの予約解除


def delete_booking_slot(request, pk):
    slot = Slot.objects.get(pk=pk)
    user = request.user
    user.assigning_slot.remove(slot)
    user.save()
    messages.success(request, "%s 登録解除しました" % slot.workname, fail_silently=True)
    return HttpResponseRedirect(reverse("shift_maker:mypage"))


def overlapping_slots(slots):
    days = slots.values_list("day", flat=True)
    days_list = list(set(days))
    overlapping_pairs = []
    for day in days_list:
        day_slots = slots.filter(day=day)
        for day_slot in day_slots:
            day_slots = day_slots.exclude(id=day_slot.id)
            for other_slot in day_slots:
                if (
                    other_slot.start_time < day_slot.end_time
                    and other_slot.end_time > day_slot.start_time
                ):
                    overlapping_pair = [day_slot.id, other_slot.id]
                    overlapping_pairs.append(overlapping_pair)
    return overlapping_pairs
