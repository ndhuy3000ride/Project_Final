from data_loader import process_audio_file, process_directory

# Xử lý một file đơn lẻ
process_audio_file(
    audio_path='Data/Data_test/Voice_10/v7test.mp3',
    output_audio_path='Processed/Voice_10/v7test_cleaned.mp3',
    output_image_dir='MelSpectrograms/Voice_10/v7test',
    segment_duration=3,
    top_db=30
)

# Hoặc xử lý toàn bộ thư mục
process_directory(
    input_dir='Data/Data_test',
    output_audio_dir='Processed',
    output_image_dir='MelSpectrograms',
    segment_duration=3,
    top_db=30
)
