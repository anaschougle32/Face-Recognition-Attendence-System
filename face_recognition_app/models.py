
from django.contrib.auth.models import User
from django.db import models

class TeacherAccount(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female')
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='teacher_profile')
    full_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    contact_number = models.CharField(max_length=15)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.full_name

class RegisteredStudent(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female')
    ]

    YEAR_CHOICES = [
        ('First Year', 'First Year'),
        ('Second Year', 'Second Year'),
        ('Third Year', 'Third Year'),
        ('Fourth Year', 'Fourth Year'),
    ]

    DIVISION_CHOICES = [
        ('Computer Science', 'Computer Science'),
        ('AI-ML', 'AI-ML'),
        ('Data Science', 'Data Science'),
        ('Artificial Intelligence', 'Artificial Intelligence'),
        ('Information Technology', 'Information Technology'),
    ]
    teacher = models.ForeignKey(TeacherAccount, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    phone_number = models.CharField(max_length=15)
    address = models.TextField()
    year = models.CharField(max_length=12, choices=YEAR_CHOICES)
    department = models.CharField(max_length=50, choices=DIVISION_CHOICES)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    age = models.IntegerField()
    date_of_birth = models.DateField()
    student_photo = models.ImageField(upload_to='student_photos/', blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.full_name

class StudentAccount(models.Model):
    student = models.OneToOneField(RegisteredStudent, on_delete=models.CASCADE, related_name='student_profile')
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student_user')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Profile for {self.student.full_name}"

class Lecture(models.Model):
    teacher = models.ForeignKey(TeacherAccount, on_delete=models.CASCADE)
    subject = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.teacher.user.username} - {self.subject} - {self.date} {self.start_time}-{self.end_time}"

class StudentAttendance(models.Model):
    student = models.ForeignKey(RegisteredStudent, on_delete=models.CASCADE)
    teacher = models.ForeignKey(TeacherAccount, on_delete=models.CASCADE)
    lecture = models.ForeignKey(Lecture, on_delete=models.CASCADE)
    datetime = models.DateTimeField()
    is_present = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.teacher.user.username} - {self.student.full_name} - {self.lecture.subject} - {self.datetime.strftime('%Y-%m-%d %H:%M')} - Present: {self.is_present}"
