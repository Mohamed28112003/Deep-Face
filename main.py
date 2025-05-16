from build_face_database import load_face_database
from find_face import find_faces
from display_results import display_results
from get_name import extract_names_from_results 

db_file = "face_db.pkl"  
face_db = load_face_database(db_file)

img_path = "test.jpg"  # Your test image

results = find_faces(
    img_path=img_path,
    face_db=face_db,
    threshold=0.6  # Adjust threshold as needed
)

# Display the results with bounding boxes and labels
display_results(
    img_path=img_path, 
    results=results, 
    confidence_threshold=0.4
)


names = extract_names_from_results(results)
print("Matched Names:", names)