from django.contrib import admin
from .models import TeacherAccount, StudentAccount,RegisteredStudent,StudentAttendance
# Register your models here.
admin.site.register(TeacherAccount)
admin.site.register(StudentAccount)
admin.site.register(RegisteredStudent)
admin.site.register(StudentAttendance)