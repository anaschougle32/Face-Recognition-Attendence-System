import base64
from datetime import timezone
import datetime
import logging
from venv import logger
import cv2
from django.forms import ValidationError
import numpy as np
from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from .models import TeacherAccount,StudentAccount,RegisteredStudent,StudentAttendance
from rest_framework import status
from rest_framework.exceptions import APIException


class TeacherSerializer(serializers.Serializer):
    full_name = serializers.CharField(max_length=100, required=True)
    email = serializers.EmailField(max_length=100, required=True)
    contact_number = serializers.CharField(max_length=15, required=True)
    gender = serializers.ChoiceField(choices=TeacherAccount.GENDER_CHOICES, required=True)
    username = serializers.CharField(max_length=100, required=True)
    password = serializers.CharField(max_length=128, required=True)
   
    def validate(self, data):
        full_name = data.get('full_name')
        email = data.get('email')
        contact_number = data.get('contact_number')

        # Check if a teacher with the same full name exists
        if TeacherAccount.objects.filter(full_name=full_name).exists():
            raise serializers.ValidationError("A teacher with this full name already exists.")

        # Check if a teacher with the same email exists
        if TeacherAccount.objects.filter(email=email).exists():
            raise serializers.ValidationError("A teacher with this email already exists.")

        # Check if a teacher with the same contact number exists
        if TeacherAccount.objects.filter(contact_number=contact_number).exists():
            raise serializers.ValidationError("A teacher with this contact number already exists.")

        return data

    def validate_password(self, value):
        if len(value) < 6:
            raise serializers.ValidationError("Password should be at least 6 characters long.")
        return value

    def create(self, validated_data):
        password = validated_data.pop('password')
        hashed_password = make_password(password)

        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email'],
            password=hashed_password
        )

        teacher = TeacherAccount.objects.create(
            user=user,
            full_name=validated_data['full_name'],
            email=validated_data['email'],
            contact_number=validated_data['contact_number'],
            gender = validated_data['gender']
            
        )
        return teacher



class RegisteredStudentSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)
    full_name = serializers.CharField(max_length=100, required=True)
    email = serializers.EmailField(required=True)
    phone_number = serializers.CharField(max_length=15, required=True)
    address = serializers.CharField(required=True)
    year = serializers.ChoiceField(choices=RegisteredStudent.YEAR_CHOICES, required=True)
    department = serializers.ChoiceField(choices=RegisteredStudent.DIVISION_CHOICES, required=True)
    gender = serializers.ChoiceField(choices=RegisteredStudent.GENDER_CHOICES, required=True)
    age = serializers.IntegerField(required=True)
    date_of_birth = serializers.DateField(required=True)
    student_photo = serializers.ImageField(required=False)  # Make it optional
    teacher = serializers.CharField(max_length=100, required=False)

    def validate(self, data):
        full_name = data.get('full_name')
        email = data.get('email')
        phone_number = data.get('phone_number')

    
        return data

    def create(self, validated_data):
        # Extracting fields similar to the Teacher model's create method
        full_name = validated_data['full_name']
        email = validated_data['email']
        phone_number = validated_data['phone_number']
        address = validated_data['address']
        year = validated_data['year']
        department = validated_data['department']
        gender = validated_data['gender']
        age = validated_data['age']
        date_of_birth = validated_data['date_of_birth']
        student_photo = validated_data.get('student_photo')
        teacher = validated_data['teacher']

        # Get the TeacherAccount instance based on the provided username
        teacher_instance = TeacherAccount.objects.get(user__username=teacher)


        # Create a new RegisteredStudent object
        return RegisteredStudent.objects.create(
            full_name=full_name,
            email=email,
            phone_number=phone_number,
            address=address,
            year=year,
            department=department,
            gender=gender,
            age=age,
            date_of_birth=date_of_birth,
            student_photo=student_photo,
            teacher=teacher_instance
        )

    def update(self, instance, validated_data):
        try:
            # Your update logic
            instance.full_name = validated_data.get('full_name', instance.full_name)
            instance.email = validated_data.get('email', instance.email)
            instance.phone_number = validated_data.get('phone_number', instance.phone_number)
            instance.address = validated_data.get('address', instance.address)
            instance.year = validated_data.get('year', instance.year)
            instance.department = validated_data.get('department', instance.department)
            instance.gender = validated_data.get('gender', instance.gender)
            instance.age = validated_data.get('age', instance.age)
            instance.date_of_birth = validated_data.get('date_of_birth', instance.date_of_birth)
            instance.student_photo = validated_data.get('student_photo', instance.student_photo)
            instance.save()
            return instance
        except Exception as e:
            # Log the detailed error information
            print(f"Error during update: {e}")
            # Raise a more specific exception, if needed
            raise APIException("Error during update. Please check the provided information.")
    



class StudentVerifySerializer(serializers.Serializer):
    full_name = serializers.CharField(max_length=100)
    email = serializers.EmailField(required=True)
    phone_number = serializers.CharField(max_length=15)
    year = serializers.ChoiceField(choices=RegisteredStudent.YEAR_CHOICES)
    department = serializers.ChoiceField(choices=RegisteredStudent.DIVISION_CHOICES)

    def validate(self, data):
        full_name = data.get('full_name')
        email = data.get('email')
        phone_number = data.get('phone_number')
        year = data.get('year')
        department = data.get('department')

        # Match the provided data to the RegisteredStudent model
        try:
            RegisteredStudent.objects.get(
                full_name=full_name,
                email=email,
                phone_number=phone_number,
                year=year,
                department=department,
            )
            return {'success': True}
        except RegisteredStudent.DoesNotExist:
            raise serializers.ValidationError("Student not found")


from rest_framework import serializers
from .models import StudentAccount, RegisteredStudent

from rest_framework import serializers

class StudentAccountSerializer(serializers.Serializer):
    email = serializers.EmailField(max_length=100, required=True)
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(max_length=128)

    def create(self, validated_data):
        email = validated_data['email']
        username = validated_data['username']
        password = validated_data['password']

        # Check if a StudentAccount with the provided email already exists
        existing_student_account = StudentAccount.objects.filter(student__email=email).first()

        if existing_student_account:
            raise serializers.ValidationError('A student account with the provided email already exists. Please use a different email.')

        try:
            # Retrieve the RegisteredStudent object by email
            registered_student = RegisteredStudent.objects.get(email=email)

            # Create User object
            user = User.objects.create_user(username=username, password=password, email=email)

            # Create StudentAccount object by associating the User and RegisteredStudent
            student_account = StudentAccount.objects.create(
                student=registered_student,
                user=user
            )

            # Return serialized data
            return {
                'success': True,
                'message': 'Student account created successfully',
                'student_account_id': student_account.id,  # Return the StudentAccount ID
                'student_username': student_account.user.username  # Return the associated username
            }
        except RegisteredStudent.DoesNotExist:
            raise serializers.ValidationError('RegisteredStudent not found with the provided email')
        except Exception as e:
            # Print detailed error information in the terminal
            print(f'Error creating student account: {str(e)}')
            raise serializers.ValidationError('Failed to create student account. Check the terminal for details.')




from rest_framework import serializers
from .models import Lecture, TeacherAccount

class LectureSerializer(serializers.Serializer):
    id = serializers.IntegerField(read_only=True)  # Set read_only to True for retrieval
    teacher = serializers.CharField(max_length=150, required=True)
    subject = serializers.CharField(max_length=100, required=True)

    def validate(self, data):
        teacher = data.get('teacher')
        subject = data.get('subject')

        # You can add additional validation logic if needed

        return data

    def create(self, validated_data):
        # Extracting fields
        teacher = validated_data['teacher']
        subject = validated_data['subject']

        # Get the TeacherAccount instance based on the provided username
        teacher_instance = TeacherAccount.objects.get(user__username=teacher)

        # Create a new Lecture object without specifying the 'id' field
        lecture = Lecture.objects.create(
            teacher=teacher_instance,
            subject=subject
        )

        # Return the serialized data excluding the 'id' field
        return {'teacher': teacher, 'subject': subject}


class GetRegisteredStudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegisteredStudent
        fields = ['full_name', 'email', 'phone_number', 'address', 'year', 'department', 'gender', 'age', 'date_of_birth', 'student_photo']


class GetLectureSerializer(serializers.ModelSerializer):
    class Meta:
        model = Lecture
        fields = ['teacher', 'subject']



class StudentAttendanceSerializer(serializers.ModelSerializer):
    student = RegisteredStudentSerializer()
    teacher = serializers.CharField(source='teacher.user.username', read_only=True)
    lecture = LectureSerializer()

    class Meta:
        model = StudentAttendance
        fields = ['id', 'student', 'teacher', 'lecture', 'datetime', 'is_present']

    def validate(self, data):
        # You can perform additional validation if needed
        return data

