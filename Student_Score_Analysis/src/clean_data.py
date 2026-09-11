import pandas as pd
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu gốc
df = pd.read_csv('../data/Exam_Score_Prediction.csv')

# Bỏ cột mã sinh viên
df_clean = df.drop(columns=['student_id'])

# Chuyển các biến dạng thứ bậc sang số
ordinal_mapping = {
    'sleep_quality': {'poor': 0, 'average': 1, 'good': 2},
    'facility_rating': {'low': 0, 'medium': 1, 'high': 2},
    'exam_difficulty': {'easy': 0, 'moderate': 1, 'hard': 2},
    'internet_access': {'no': 0, 'yes': 1}
}

for col, mapping in ordinal_mapping.items():
    df_clean[col] = df_clean[col].map(mapping)

# Chuyển biến dạng phân loại thành các cột 0/1
nominal_cols = ['gender', 'course', 'study_method']

df_encoded = pd.get_dummies(
    df_clean,
    columns=nominal_cols,
    drop_first=True
)

# Chuẩn hóa các biến số
scaler = StandardScaler()

scale_cols = [
    'age',
    'study_hours',
    'class_attendance',
    'sleep_hours'
]

df_encoded[scale_cols] = scaler.fit_transform(
    df_encoded[scale_cols]
)

# Lưu dữ liệu sau khi xử lý
df_encoded.to_csv(
    '../data/Exam_Score_Prediction_Cleaned.csv',
    index=False
)

print("-> Thành công! File 'Exam_Score_Prediction_Cleaned.csv' đã xuất hiện trong thư mục.")