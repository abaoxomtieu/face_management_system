"""
Simple Streamlit UI for Face Recognition API Testing
"""

import streamlit as st
import requests
import json
from io import BytesIO
import time

# Configuration
API_BASE_URL = "http://localhost:9001"

def main():
    st.set_page_config(
        page_title="Face Recognition API Tester",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Face Recognition API Tester")
    st.markdown("Simple interface to test all Face Recognition API endpoints")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Home", "🏫 Classes", "👥 Students", "✅ Attendance", "🔧 System"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # Quick stats
        try:
            health_response = requests.get(f"{API_BASE_URL}/api/face-recognition/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success("🟢 API Online")
                st.metric("Students", health_data.get('total_students', 0))
                st.metric("Model", "✅ Loaded" if health_data.get('model_loaded', False) else "❌ Not Loaded")
            else:
                st.error("🔴 API Offline")
        except:
            st.warning("⚠️ Cannot connect to API")
    
    # Main content
    if page == "🏠 Home":
        render_home()
    elif page == "🏫 Classes":
        render_classes()
    elif page == "👥 Students":
        render_students()
    elif page == "✅ Attendance":
        render_attendance()
    elif page == "🔧 System":
        render_system()

def render_home():
    """Home page with API overview"""
    st.header("🏠 API Overview")
    
    # API Status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌐 API Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Server: Online")
            else:
                st.error("❌ API Server: Error")
        except:
            st.error("❌ API Server: Offline")
    
    with col2:
        st.subheader("🎭 Face Recognition")
        try:
            response = requests.get(f"{API_BASE_URL}/api/face-recognition/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Face Recognition: Online")
                st.write(f"Model: {data.get('model_name', 'N/A')}")
            else:
                st.error("❌ Face Recognition: Error")
        except:
            st.error("❌ Face Recognition: Offline")
    
    with col3:
        st.subheader("📚 Documentation")
        st.markdown(f"[Swagger UI]({API_BASE_URL}/)")
        st.markdown(f"[Health Check]({API_BASE_URL}/health)")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View System Stats", use_container_width=True):
            st.switch_page("🔧 System")
    
    with col2:
        if st.button("🏫 Manage Classes", use_container_width=True):
            st.switch_page("🏫 Classes")
    
    with col3:
        if st.button("👥 Manage Students", use_container_width=True):
            st.switch_page("👥 Students")

def render_classes():
    """Class management page"""
    st.header("🏫 Class Management")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 View Classes", "➕ Create Class", "🗑️ Delete Class"])
    
    with tab1:
        st.subheader("📋 All Classes")
        
        try:
            response = requests.get(f"{API_BASE_URL}/api/classes", timeout=5)
            if response.status_code == 200:
                data = response.json()
                classes = data.get('classes', [])
                
                if classes:
                    st.success(f"Found {len(classes)} classes")
                    
                    for class_info in classes:
                        with st.expander(f"🏫 {class_info.get('class_id', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Class ID:** {class_info.get('class_id', 'N/A')}")
                            with col2:
                                st.metric("Students", class_info.get('total_students', 0))
                else:
                    st.info("No classes found")
            else:
                st.error(f"Failed to load classes: {response.status_code}")
        except Exception as e:
            st.error(f"Error loading classes: {e}")
    
    with tab2:
        st.subheader("➕ Create New Class")
        
        with st.form("create_class_form"):
            class_id = st.text_input("Class ID *", placeholder="e.g., 10A1")
            
            submitted = st.form_submit_button("Create Class", type="primary")
            
            if submitted:
                if not class_id:
                    st.warning("Please fill in Class ID")
                else:
                    try:
                        data = {
                            "class_id": class_id
                        }
                        
                        response = requests.post(f"{API_BASE_URL}/api/classes", json=data, timeout=5)
                        
                        if response.status_code == 200:
                            st.success("✅ Class created successfully!")
                            try:
                                st.toast("Class created", icon="✅")
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to create class: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    with tab3:
        st.subheader("🗑️ Delete Class")
        
        # Get classes for selection
        try:
            response = requests.get(f"{API_BASE_URL}/api/classes", timeout=5)
            if response.status_code == 200:
                data = response.json()
                classes = data.get('classes', [])
                
                if classes:
                    class_ids = [cls.get('class_id') for cls in classes]
                    selected_class = st.selectbox("Select Class to Delete", class_ids)
                    
                    if selected_class:
                        st.warning("⚠️ This will permanently delete the class and all its students.")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Delete Class", type="primary", use_container_width=True):
                                try:
                                    del_resp = requests.delete(f"{API_BASE_URL}/api/classes/{selected_class}", timeout=15)
                                    if del_resp.status_code == 200:
                                        st.success("✅ Class deleted successfully!")
                                        try:
                                            st.toast("Class deleted", icon="✅")
                                        except Exception:
                                            pass
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete class: {del_resp.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        with col2:
                            st.info("Action is irreversible")
                else:
                    st.info("No classes found")
        except Exception as e:
            st.error(f"Error loading classes: {e}")

def render_students():
    """Student management page"""
    st.header("👥 Student Management")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 View Students", "➕ Register Student", "✏️ Update Student", "🗑️ Delete Student"])
    
    with tab1:
        st.subheader("📋 All Students")
        
        # Filter by class
        try:
            # Get classes for filter
            class_response = requests.get(f"{API_BASE_URL}/api/classes", timeout=5)
            if class_response.status_code == 200:
                class_data = class_response.json()
                classes = class_data.get('classes', [])
                
                if classes:
                    class_options = ["All Classes"] + [cls.get('class_id') for cls in classes]
                    selected_class_filter = st.selectbox("Filter by Class", class_options)
                    
                    # Determine class_id for filter
                    class_id_filter = None
                    if selected_class_filter != "All Classes":
                        class_id_filter = selected_class_filter
                else:
                    class_id_filter = None
            else:
                class_id_filter = None
        except:
            class_id_filter = None
        
        try:
            # Get students with optional filter
            if class_id_filter:
                response = requests.get(f"{API_BASE_URL}/api/face-recognition/students?class_id={class_id_filter}", timeout=5)
            else:
                response = requests.get(f"{API_BASE_URL}/api/face-recognition/students", timeout=5)
                
            if response.status_code == 200:
                data = response.json()
                students = data.get('students', [])
                
                if students:
                    filter_text = f" in class {class_id_filter}" if class_id_filter else ""
                    st.success(f"Found {len(students)} students{filter_text}")
                    
                    for student in students:
                        with st.expander(f"👤 Student {student.get('student_id', 'N/A')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Student ID:** {student.get('student_id', 'N/A')}")
                                st.write(f"**Class ID:** {student.get('class_id', 'N/A')}")
                            with col2:
                                st.metric("Face Images", student.get('num_images', 0))
                else:
                    filter_text = f" in class {class_id_filter}" if class_id_filter else ""
                    st.info(f"No students found{filter_text}")
            else:
                st.error(f"Failed to load students: {response.status_code}")
        except Exception as e:
            st.error(f"Error loading students: {e}")
    
    with tab2:
        st.subheader("➕ Register New Student")
        
        # Get classes for selection
        try:
            response = requests.get(f"{API_BASE_URL}/api/classes", timeout=5)
            if response.status_code == 200:
                data = response.json()
                classes = data.get('classes', [])
                
                if classes:
                    class_options = {cls.get('class_id'): cls.get('class_id') for cls in classes}
                    
                    with st.form("register_student_form"):
                        student_id = st.text_input("Student ID *", placeholder="e.g., SV001")
                        selected_class = st.selectbox("Class *", list(class_options.keys()))
                        images = st.file_uploader(
                            "Face Images *",
                            type=['jpg', 'jpeg', 'png'],
                            accept_multiple_files=True,
                            help="Upload 3-5 clear face images from different angles"
                        )
                        
                        submitted = st.form_submit_button("Register Student", type="primary")
                        
                        if submitted:
                            if not student_id or not selected_class or not images:
                                st.warning("Please fill in all required fields and upload images")
                            else:
                                try:
                                    class_id = class_options[selected_class]
                                    
                                    files = []
                                    for i, image in enumerate(images):
                                        files.append(('images', (f'image_{i+1}.jpg', image.getvalue(), image.type)))
                                    
                                    data = {
                                        'student_id': student_id,
                                        'class_id': class_id
                                    }
                                    
                                    response = requests.post(
                                        f"{API_BASE_URL}/api/face-recognition/students/register",
                                        files=files,
                                        data=data,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success("✅ Student registered successfully!")
                                        if result.get('student'):
                                            student_info = result['student']
                                            st.write(f"Student ID: {student_info.get('student_id', 'N/A')}")
                                            st.write(f"Class ID: {student_info.get('class_id', 'N/A')}")
                                            st.write(f"Images: {student_info.get('num_images', 0)}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to register student: {response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                                    import traceback
                                    st.error(f"Traceback: {traceback.format_exc()}")
                else:
                    st.warning("No classes found. Please create a class first.")
        except Exception as e:
            st.error(f"Error loading classes: {e}")
    
    with tab3:
        st.subheader("✏️ Update Student")
        
        # Get students for selection
        try:
            response = requests.get(f"{API_BASE_URL}/api/face-recognition/students", timeout=5)
            if response.status_code == 200:
                data = response.json()
                students = data.get('students', [])
                
                if students:
                    student_options = {f"{std.get('student_id')} ({std.get('class_id')})": std.get('student_id') for std in students}
                    selected_student = st.selectbox("Select Student to Update", list(student_options.keys()))
                    
                    if selected_student:
                        student_id = student_options[selected_student]
                        
                        with st.form("update_student_form"):
                            # Get classes for selection
                            try:
                                class_response = requests.get(f"{API_BASE_URL}/api/classes", timeout=5)
                                if class_response.status_code == 200:
                                    class_data = class_response.json()
                                    classes = class_data.get('classes', [])
                                    
                                    if classes:
                                        class_options = {cls.get('class_id'): cls.get('class_id') for cls in classes}
                                        new_class_id = st.selectbox("New Class ID", ["Keep Current"] + list(class_options.keys()))
                                        
                                        if new_class_id == "Keep Current":
                                            new_class_id = None
                                        else:
                                            new_class_id = new_class_id
                                    else:
                                        new_class_id = st.text_input("New Class ID", placeholder="Enter new class ID")
                                else:
                                    new_class_id = st.text_input("New Class ID", placeholder="Enter new class ID")
                            except:
                                new_class_id = st.text_input("New Class ID", placeholder="Enter new class ID")
                            
                            new_images = st.file_uploader(
                                "New Face Images",
                                type=['jpg', 'jpeg', 'png'],
                                accept_multiple_files=True,
                                help="Upload new face images (optional)"
                            )
                            
                            submitted = st.form_submit_button("Update Student", type="primary")
                            
                            if submitted:
                                try:
                                    data = {}
                                    if new_class_id:
                                        data['class_id'] = new_class_id
                                    
                                    files = []
                                    if new_images:
                                        for image in new_images:
                                            files.append(('images', image.getvalue(), image.type))
                                    
                                    response = requests.put(
                                        f"{API_BASE_URL}/api/face-recognition/students/{student_id}",
                                        files=files,
                                        data=data,
                                        timeout=30
                                    )
                                    
                                    if response.status_code == 200:
                                        st.success("✅ Student updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to update student: {response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                else:
                    st.info("No students found")
        except Exception as e:
            st.error(f"Error loading students: {e}")
    
    with tab4:
        st.subheader("🗑️ Delete Student")
        
        # Get students for selection
        try:
            response = requests.get(f"{API_BASE_URL}/api/face-recognition/students", timeout=5)
            if response.status_code == 200:
                data = response.json()
                students = data.get('students', [])
                
                if students:
                    student_options = {f"{std.get('student_id')} ({std.get('class_id')})": std.get('student_id') for std in students}
                    selected_student = st.selectbox("Select Student to Delete", list(student_options.keys()))
                    
                    if selected_student:
                        student_id = student_options[selected_student]
                        
                        # Show student info
                        st.warning("⚠️ This action will permanently delete:")
                        st.write(f"- Student ID: {student_id}")
                        st.write("- All face images")
                        st.write("- All embeddings from database")
                        st.write("- Student metadata")
                        
                        # Confirmation
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Delete Student", type="primary", use_container_width=True):
                                try:
                                    response = requests.delete(f"{API_BASE_URL}/api/face-recognition/students/{student_id}", timeout=10)
                                    
                                    if response.status_code == 200:
                                        result = response.json()
                                        st.success("✅ Student deleted successfully!")
                                        st.write(f"Message: {result.get('message', 'N/A')}")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete student: {response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                        
                        with col2:
                            if st.button("❌ Cancel", use_container_width=True):
                                st.info("Deletion cancelled")
                else:
                    st.info("No students found")
        except Exception as e:
            st.error(f"Error loading students: {e}")

def render_attendance():
    """Attendance testing page"""
    st.header("✅ Attendance Testing")
    
    st.subheader("🎭 Face Recognition Test")
    
    # Get classes for selection
    try:
        response = requests.get(f"{API_BASE_URL}/api/classes", timeout=5)
        if response.status_code == 200:
            data = response.json()
            classes = data.get('classes', [])
            
            if classes:
                class_options = {cls.get('class_id'): cls.get('class_id') for cls in classes}
                selected_class = st.selectbox("Select Class", list(class_options.keys()))
                
                # Image upload
                uploaded_file = st.file_uploader(
                    "Upload Face Image",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload an image containing a single student's face"
                )
                
                if uploaded_file:
                    st.image(uploaded_file, width=300, caption="Uploaded Image")
                    
                    if st.button("🔍 Check Attendance", type="primary"):
                        with st.spinner("Processing attendance..."):
                            try:
                                class_id = class_options[selected_class]
                                
                                files = {'image': uploaded_file.getvalue()}
                                data = {'class_id': class_id}
                                
                                response = requests.post(
                                    f"{API_BASE_URL}/api/face-recognition/attendance",
                                    files=files,
                                    data=data,
                                    timeout=10
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    
                                    if result.get('match_found'):
                                        st.success("✅ Student Recognized!")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**Student ID:** {result.get('student_id', 'N/A')}")
                                            st.write(f"**Class ID:** {result.get('class_id', 'N/A')}")
                                        
                                        with col2:
                                            st.metric("Confidence", f"{result.get('confidence', 0):.3f}")
                                            st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                                    else:
                                        st.warning("❌ No matching student found in this class")
                                        st.write(f"Processing time: {result.get('processing_time', 0):.2f}s")
                                else:
                                    st.error(f"❌ API Error: {response.status_code}")
                                    st.write(f"Response: {response.text}")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
            else:
                st.warning("No classes found. Please create a class first.")
        else:
            st.error("Failed to load classes")
    except Exception as e:
        st.error(f"Error loading classes: {e}")

def render_system():
    """System monitoring page"""
    st.header("🔧 System Monitoring")
    
    # Health Check
    st.subheader("🏥 Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/face-recognition/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.success("✅ System is healthy!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", data.get('status', 'Unknown'))
            with col2:
                st.metric("Model Loaded", "✅ Yes" if data.get('model_loaded', False) else "❌ No")
            with col3:
                st.metric("Total Students", data.get('total_students', 0))
            with col4:
                st.metric("Embedding Dimension", data.get('embedding_dimension', 0))
        else:
            st.error(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Health check error: {e}")
    
    # System Statistics
    st.subheader("📊 System Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/face-recognition/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            stats = data.get('stats', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Database Statistics:**")
                st.write(f"- Total Students: {stats.get('total_students', 0)}")
                st.write(f"- Total Embeddings: {stats.get('total_embeddings', 0)}")
                st.write(f"- Embedding Dimension: {stats.get('embedding_dimension', 0)}")
                st.write(f"- Index Type: {stats.get('index_type', 'N/A')}")
            
            with col2:
                st.write("**Model Configuration:**")
                st.write(f"- Model Name: {stats.get('model_name', 'N/A')}")
                st.write(f"- Detection Threshold: {stats.get('detection_threshold', 0):.2f}")
                st.write(f"- Recognition Threshold: {stats.get('recognition_threshold', 0):.2f}")
                st.write(f"- Model Loaded: {'Yes' if stats.get('model_loaded', False) else 'No'}")
        else:
            st.error(f"❌ Failed to load stats: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Stats error: {e}")
    
    # System Actions
    st.subheader("⚙️ System Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding index..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/api/face-recognition/rebuild-index", timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Index rebuilt successfully!")
                        st.write(f"Total embeddings: {data.get('total_embeddings', 0)}")
                        st.write(f"Processing time: {data.get('processing_time', 0):.2f}s")
                    else:
                        st.error(f"❌ Failed to rebuild index: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()

if __name__ == "__main__":
    main()
