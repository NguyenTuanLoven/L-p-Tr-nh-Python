import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu gốc
df = pd.read_csv('../data/Exam_Score_Prediction.csv')

print("Kích thước dữ liệu:")
print(df.shape)

print("\nTên các cột:")
print(df.columns.tolist())

print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())
print("\nKiểm tra dữ liệu thiếu:")
print(df.isnull().sum())
print("\nSố dòng bị trùng:")
print(df.duplicated().sum())
print("\n===== THỐNG KÊ ĐIỂM THI =====")

print("Điểm trung bình:",
      df['exam_score'].mean())

print("Điểm cao nhất:",
      df['exam_score'].max())

print("Điểm thấp nhất:",
      df['exam_score'].min())

print("Độ lệch chuẩn:",
      df['exam_score'].std())
print("\n===== PHÂN TÍCH THỜI GIAN HỌC =====")

study_result = df.groupby('study_hours')['exam_score'].mean()

print("Điểm trung bình theo số giờ học:")
print(study_result)
print("\n===== NHÓM THỜI GIAN HỌC =====")

df['study_hours_group'] = pd.cut(
    df['study_hours'],
    bins=[0, 2, 4, 6, 8],
    labels=['0-2 giờ', '2-4 giờ', '4-6 giờ', '6-8 giờ']
)

study_group_result = df.groupby(
    'study_hours_group',
    observed=False
)['exam_score'].mean()

print("Điểm trung bình theo nhóm thời gian học:")
print(study_group_result)
# Vẽ biểu đồ điểm trung bình theo nhóm thời gian học
plt.figure(figsize=(8, 5))

study_group_result.plot(kind='bar')

plt.title('Điểm thi trung bình theo nhóm thời gian học')
plt.xlabel('Nhóm thời gian học')
plt.ylabel('Điểm thi trung bình')

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig('../result/study_hours_analysis.png')

plt.show()