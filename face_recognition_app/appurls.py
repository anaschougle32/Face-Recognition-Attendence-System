from django.urls import path
from . import views
from .views import RegisteredStudentListCreateView,GetAllAttendanceView,LectureListCreateView,SearchAttendanceView,SearchStudentView,GetStudentAttendanceView,SearchStudentAttendanceView,ExportStudentAttendanceCSV

urlpatterns = [
    path('login/', views.login, name='login'),
    path('signup_teacher/', views.signup_teacher, name='signup_teacher'),
    path('verify_student/', views.verify_student, name='verify_student'),
    path('create_student_credentials/', views.create_student_credentials, name='create_student_credentials'),

    path('registered-students/', RegisteredStudentListCreateView.as_view(), name='registered-students'),
    path('delete-registered-students/<int:pk>/', RegisteredStudentListCreateView.as_view(), name='Delete-registered-students'),
    path('update-registered-students/<int:pk>/', RegisteredStudentListCreateView.as_view(), name='update-registered-students'),
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('search-student/', SearchStudentView.as_view(), name='search_student'),
    path('student_attendance/', GetStudentAttendanceView.as_view(), name='student-attendance-detail'),
    path('search-student-attendance/', SearchStudentAttendanceView.as_view(), name='search-student-attendance'),

    path('get_all_attendance/', GetAllAttendanceView.as_view(), name='all_attendance'),
    path('delete_attendance/<int:pk>/', GetAllAttendanceView.as_view(), name='delete_attendance'),
    path('search-attendance/', SearchAttendanceView.as_view(), name='search_attendance'),

    path('lectures/', LectureListCreateView.as_view(), name='lecture-list-create'),
    path('delete_lecture/<int:pk>/', views.LectureListCreateView.as_view(), name='delete_lecture'),
  # New URL for exporting student attendance in CSV
    path('export-student-attendance-csv/', ExportStudentAttendanceCSV.as_view(), name='export_student_attendance_csv'),

]



