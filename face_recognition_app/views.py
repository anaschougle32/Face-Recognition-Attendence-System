from datetime import timezone, datetime
import logging
import cv2
import numpy as np
from django.contrib.auth import authenticate
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import (
    RegisteredStudent,
    StudentAccount,
    TeacherAccount,
    StudentAttendance,
    Lecture,
)
from .serializers import (
    TeacherSerializer,
    StudentVerifySerializer,
    StudentAccountSerializer,
    RegisteredStudentSerializer,
    LectureSerializer,
    StudentAttendanceSerializer,
)

from scipy.spatial import distance
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.utils import timezone
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.core.exceptions import ValidationError
from rest_framework.generics import (
    ListAPIView,
    RetrieveUpdateDestroyAPIView,
    CreateAPIView,
)
import face_recognition
from rest_framework.views import APIView
from face_recognition_app import serializers
from rest_framework.generics import RetrieveAPIView
from rest_framework.filters import SearchFilter


@api_view(['POST'])

def mark_attendance(request):
    try:
        print(request.data)
        # Get the received face data from the request
        unknown_face_image = request.FILES.get('face_data')

        # Check if face data is provided
        if not unknown_face_image:
            print("Error: Face data not provided")
            return Response({'error': 'Face data not provided'}, status=400)

        # Save the received face data temporarily
        temp_image_path = default_storage.save('temp_unknown_face_image.jpeg', ContentFile(unknown_face_image.read()))
        print(f"Success: Saved face data at {temp_image_path}")

        # Load the unknown face image
        unknown_face_img = face_recognition.load_image_file(temp_image_path)
        print("Success: Loaded unknown face image")

        # Detect faces in the unknown face image using face_recognition
        face_locations = face_recognition.face_locations(unknown_face_img)
        print("Success: Detected faces using face_recognition")

        # If no face is found using face_recognition, try Haar cascades
        if not face_locations:
            face_locations = detect_faces_with_haar_cascades(unknown_face_img)
            print("Success: Detected faces using Haar cascades")

        if not face_locations:
            # No face found in the provided image
            print("Error: No face found in the provided image")
            return Response({'error': 'No face found in the provided image'}, status=400)

        unknown_face_encoding = face_recognition.face_encodings(unknown_face_img, known_face_locations=face_locations)[0]
        print("Success: Encoded unknown face")

        # Process each registered student
        recognized_students = []

        for student in RegisteredStudent.objects.all():
            # Load the student's photo
            student_photo = face_recognition.load_image_file(student.student_photo.path)
            print(f"Success: Loaded {student.full_name}'s photo")

            # Detect faces in the student's photo using face_recognition
            student_face_locations = face_recognition.face_locations(student_photo)
            print(f"Success: Detected faces in {student.full_name}'s photo")

            # If no face is found using face_recognition, try Haar cascades
            if not student_face_locations:
                student_face_locations = detect_faces_with_haar_cascades(student_photo)
                print(f"Success: Detected faces in {student.full_name}'s photo using Haar cascades")

            if not student_face_locations:
                # Skip students with no face found in their photo
                print(f"Skipped: No face found in {student.full_name}'s photo")
                continue

            # Encode the first face found in the student's photo
            student_face_encoding = face_recognition.face_encodings(student_photo, known_face_locations=student_face_locations)[0]
            print(f"Success: Encoded {student.full_name}'s face")

            # Compare the unknown face encoding with the student's face encoding using Euclidean distance
            face_distance = distance.euclidean(student_face_encoding, unknown_face_encoding)
            print(f"Face distance for {student.full_name}: {face_distance}")

            # Set a threshold for considering a match (you may need to adjust this value)
            match_threshold = 0.6

            if face_distance < match_threshold:
                # Check if attendance is already marked for today
                today = timezone.now().date()
                if StudentAttendance.objects.filter(student=student, datetime__date=today).exists():
                    default_storage.delete(temp_image_path)
                    print(f"Error: Attendance already marked for {student.full_name} today")
                    return Response({'error': f'Attendance already marked for {student.full_name} today'}, status=400)

                try:
                    # Access the StudentAccount associated with the matched RegisteredStudent
                    student_account = student
                    print(student)
                    # Get the teacher account based on the provided username
                    teacher_username = request.data.get('teacher_username', '')
                    teacher_account = get_object_or_404(TeacherAccount, user__username=teacher_username)

                    subject_name = request.data.get('subject', '')
                    lecture = get_object_or_404(Lecture, teacher=teacher_account,subject=subject_name)

                    # Pass both StudentAccount, TeacherAccount, and Lecture to the update_or_create_attendance_objects function
                    update_or_create_attendance_objects(student_account, teacher_account, lecture)

                    recognized_students.append(student.id)
                except StudentAccount.DoesNotExist:
                    print(f"StudentAccount not found for {student.full_name}")



        # Remove the temporary image file
        default_storage.delete(temp_image_path)

        return Response({'recognized_students': recognized_students})
    
    except Exception as e:
        print(f"An error occurred: {e}")
        default_storage.delete(temp_image_path)
        return Response({'error': 'An error occurred'}, status=500)



def detect_faces_with_haar_cascades(image):
    # Convert the image to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load Haar cascades for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces using Haar cascades
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Filter out false positives based on aspect ratio and size
    filtered_faces = []
    for (x, y, w, h) in faces:
        aspect_ratio = w / h
        if 0.5 < aspect_ratio < 1.5 and 50 < w < 300 and 50 < h < 300:
            filtered_faces.append((y, x + w, y + h, x))

    return filtered_faces



def update_or_create_attendance_objects(student_account, teacher_account, lecture):
    current_datetime = timezone.now()

    attendance_object, created = StudentAttendance.objects.get_or_create(
        student=student_account,
        teacher=teacher_account,
        lecture=lecture,
        datetime=current_datetime,
        defaults={'is_present': True}
    )

    if not created:
        attendance_object.is_present = True
        attendance_object.save()


@api_view(['POST'])
def signup_teacher(request):
    serializer = TeacherSerializer(data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response({"message": "Teacher registered successfully"}, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def verify_student(request):
    serializer = StudentVerifySerializer(data=request.data)

    if serializer.is_valid():
        # Student data is valid, indicating a match in the RegisteredStudent model
        return Response({"message": "Student found in Registered Students"}, status=status.HTTP_200_OK)

    # If the data provided does not match any student in the RegisteredStudent model
    return Response(serializer.errors, status=status.HTTP_404_NOT_FOUND)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import StudentAccountSerializer

@api_view(['POST'])
def create_student_credentials(request):
    serializer = StudentAccountSerializer(data=request.data)

    try:
        serializer.is_valid(raise_exception=True)  # Raise an exception for validation errors

        # If validation passes, save the serializer data
        result = serializer.save()
        return Response(result, status=status.HTTP_201_CREATED)

    except serializers.ValidationError as validation_error:
        # Extract the error message from the validation error
        error_message = str(validation_error.detail[0]) if validation_error.detail else 'Invalid data.'

        return Response({'error': error_message}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')

    user = authenticate(username=username, password=password)

    if user:
        if hasattr(user, 'teacher_profile'):
            return Response({"message": "Login successful as a teacher", "user_type": "teacher"}, status=status.HTTP_200_OK)
        elif hasattr(user, 'student_user'):
            return Response({"message": "Login successful as a student", "user_type": "student"}, status=status.HTTP_200_OK)

    return Response({"message": "User type not recognized"}, status=status.HTTP_401_UNAUTHORIZED)


class RegisteredStudentListCreateView(APIView):
    def get(self, request):
        teacher_username = request.query_params.get('teacher', None)
        if teacher_username:
            students = RegisteredStudent.objects.filter(teacher__user__username=teacher_username)
        else:
            students = RegisteredStudent.objects.all()

        serializer = RegisteredStudentSerializer(students, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = RegisteredStudentSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

   
    def put(self, request, pk):
        student = get_object_or_404(RegisteredStudent, pk=pk)
        serializer = RegisteredStudentSerializer(student, data=request.data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        student = RegisteredStudent.objects.get(pk=pk)
        student.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
  



class LectureListCreateView(APIView):
    serializer_class = LectureSerializer

    def get(self, request):
        teacher_username = self.request.query_params.get('teacher_username')
        lectures = Lecture.objects.filter(teacher__user__username=teacher_username)
        serializer = self.serializer_class(lectures, many=True)
        return Response(serializer.data)


    def post(self, request):
        serializer = self.serializer_class(data=request.data, context={'request': request})
        try:
            serializer.is_valid(raise_exception=True)
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except ValidationError as e:
            # Log detailed information about the validation error
            print(f"Validation Error: {e}")
            return Response({"error": "Invalid data. Check the input and try again."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            # Log detailed information about the unexpected error
            print(f"Unexpected Error: {e}")
            return Response({"error": "An unexpected error occurred. Please try again later."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    def delete(self, request, pk):
        try:
            lecture = Lecture.objects.get(pk=pk)
            lecture.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Lecture.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)



class GetAllAttendanceView(ListAPIView):
    serializer_class = StudentAttendanceSerializer

    def get_queryset(self):
        teacher_username = self.request.query_params.get('teacher', None)
        if teacher_username:
            return StudentAttendance.objects.filter(teacher__user__username=teacher_username)
        else:
            return StudentAttendance.objects.all()


    def delete(self, request, pk):
        try:
            attendance = StudentAttendance.objects.get(pk=pk)
            attendance.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Lecture.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)



class SearchAttendanceView(ListAPIView):
    serializer_class = StudentAttendanceSerializer

    def get_queryset(self):
        teacher_username = self.request.query_params.get('teacher', None)
        query = self.request.query_params.get('query', None)

        if teacher_username and query:
            # Search for student attendance based on the query and teacher
            return StudentAttendance.objects.filter(
                teacher__user__username=teacher_username,
                student__full_name__icontains=query
            )
        else:
            return StudentAttendance.objects.none()
        


class SearchStudentView(ListAPIView):
    serializer_class = RegisteredStudentSerializer

    def get_queryset(self):
        teacher_username = self.request.query_params.get('teacher', None)
        query = self.request.query_params.get('query', None)

        if teacher_username and query:
            # Search for student attendance based on the query and teacher
            return RegisteredStudent.objects.filter(
                teacher__user__username=teacher_username,
                full_name__icontains=query
            )
        else:
            return RegisteredStudent.objects.none()
        

class GetStudentAttendanceView(ListAPIView):
    serializer_class = StudentAttendanceSerializer

    def get_queryset(self):
        student_username = self.request.query_params.get('student', None)

        if student_username:
            try:
                # Get the StudentAccount based on the student username
                student_account = StudentAccount.objects.get(user__username=student_username)

                # Get the RegisteredStudent associated with the StudentAccount
                registered_student = student_account.student

                # Filter StudentAttendance based on the RegisteredStudent
                queryset = StudentAttendance.objects.filter(student=registered_student)
                return queryset
            except StudentAccount.DoesNotExist:
                return StudentAttendance.objects.none()
        else:
            print("Error in getting the student's attendances")

class SearchStudentAttendanceView(ListAPIView):
    serializer_class = StudentAttendanceSerializer

    def get_queryset(self):
        student_username = self.request.query_params.get('student', None)
        query = self.request.query_params.get('query', None)

        if student_username and query:
            # Get the StudentAccount based on the student username
            student_account = StudentAccount.objects.get(user__username=student_username)

            # Get the RegisteredStudent associated with the StudentAccount
            registered_student = student_account.student

            # Search for student attendance based on the query and student
            queryset = StudentAttendance.objects.filter(
                student=registered_student,
                lecture__subject__icontains=query
            )

            return queryset
        else:
            return StudentAttendance.objects.none()







import csv
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import RegisteredStudent, StudentAttendance, Lecture
from .serializers import RegisteredStudentSerializer
import logging

logger = logging.getLogger(__name__)


class ExportStudentAttendanceCSV(APIView):
    def get(self, request, *args, **kwargs):
        teacher_username = request.query_params.get('teacher', None)

        try:
            if teacher_username:
                students = RegisteredStudent.objects.filter(
                    teacher__user__username=teacher_username,
                )

                # Create the CSV response
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{teacher_username}_student_attendance.csv"'

                # Create a CSV writer
                writer = csv.writer(response)

                # Write the header
                writer.writerow(['Student ID', 'Full Name', 'Year', 'Department', 'Subject'])

                # Write the data rows
                for student in students:
                    # Get the attendance data for the student
                    student_attendance = StudentAttendance.objects.filter(student=student)
                    
                    # Calculate total lectures attended for the student
                    total_lectures_attended = student_attendance.count()

                    # Serialize the student data
                    student_data = RegisteredStudentSerializer(student).data

                    # Write the data rows
                    for attendance in student_attendance:
                        lecture_subject = attendance.lecture.subject
                        writer.writerow([
                            student_data['id'],
                            student_data['full_name'],
                            student_data['year'],
                            student_data['department'],
                            lecture_subject,
                        ])

                return response

            return Response({'detail': 'Invalid parameters for exporting student attendance.'}, status=400)
        except Exception as e:
            logger.error(f"An error occurred while exporting student attendance: {str(e)}")
            return Response({'detail': 'An error occurred while exporting student attendance.'}, status=500)
            
    def get_subject_and_lectures_attended(self, student, teacher_username):
        subjects_attended = []

        try:
            # Get the lectures attended by the student for each subject
            subjects = Lecture.objects.filter(teacher__user__username=teacher_username)
            for subject in subjects:
                lectures_attended = StudentAttendance.objects.filter(
                    student=student,
                    lecture__teacher__user__username=teacher_username,
                    lecture__subject=subject.subject
                ).count()

                subjects_attended.append(f"{subject.subject}: {lectures_attended} lectures")

            return ', '.join(subjects_attended)
        except Exception as e:
            logger.error(f"An error occurred while getting subject and lectures attended: {str(e)}")
            return ''





































