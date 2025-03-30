import torchvision
import os
from torchvision.datasets import VOCDetection, VOCSegmentation

def download_voc_with_torchvision(root="../datasets", year="2012", download=True):
    """
    Torchvision을 사용하여 PASCAL VOC 데이터셋을 다운로드합니다.
    
    Args:
        root: 데이터셋 저장 경로
        year: 데이터셋 연도 (2007 또는 2012)
        download: 다운로드 여부
    """
    print(f"===== VOC{year} 객체 탐지 데이터셋 다운로드 =====")
    
    # 객체 탐지(Detection) 데이터셋 다운로드
    detection_dataset = VOCDetection(
        root=root, 
        year=year, 
        image_set='trainval', 
        download=download
    )
    
    print(f"\n데이터셋 크기: {len(detection_dataset)} 이미지")
    print(f"첫 번째 이미지 크기: {detection_dataset[0][0].size}")
    print(f"첫 번째 이미지 어노테이션 예시: {list(detection_dataset[0][1]['annotation'].keys())}")
    
    print(f"\n===== VOC{year} 세그멘테이션 데이터셋 다운로드 =====")
    
    # 세그멘테이션(Segmentation) 데이터셋 다운로드
    try:
        segmentation_dataset = VOCSegmentation(
            root=root, 
            year=year, 
            image_set='trainval', 
            download=download
        )
        
        print(f"\n데이터셋 크기: {len(segmentation_dataset)} 이미지")
        print(f"첫 번째 이미지 크기: {segmentation_dataset[0][0].size}")
        print(f"첫 번째 세그멘테이션 마스크 크기: {segmentation_dataset[0][1].size}")
    except Exception as e:
        print(f"세그멘테이션 데이터셋 다운로드 오류: {e}")
    
    return detection_dataset, segmentation_dataset

def check_dataset_structure(root="../datasets"):
    """
    다운로드된 데이터셋의 디렉토리 구조를 확인합니다.
    """
    voc_dir = os.path.join(root, "VOCdevkit")
    
    if not os.path.exists(voc_dir):
        print(f"VOCdevkit 디렉토리를 찾을 수 없습니다: {voc_dir}")
        return
    
    print("\n===== 데이터셋 디렉토리 구조 =====")
    
    for year_dir in os.listdir(voc_dir):
        if year_dir.startswith("VOC"):
            year_path = os.path.join(voc_dir, year_dir)
            print(f"\n{year_dir}:")
            
            subdirs = os.listdir(year_path)
            for subdir in subdirs:
                subdir_path = os.path.join(year_path, subdir)
                if os.path.isdir(subdir_path):
                    num_files = len(os.listdir(subdir_path))
                    print(f"  - {subdir}: {num_files}개 파일")

def main():
    # 데이터셋 저장 경로
    root = "../datasets"
    os.makedirs(root, exist_ok=True)
    
    # VOC2007 다운로드
    print("\n===== VOC2007 다운로드 중 =====")
    download_voc_with_torchvision(root=root, year="2007", download=True)
    
    # VOC2012 다운로드
    print("\n===== VOC2012 다운로드 중 =====")
    download_voc_with_torchvision(root=root, year="2012", download=True)
    
    # 데이터셋 구조 확인
    check_dataset_structure(root)
    
    print("\n다운로드가 완료되었습니다!")
    print(f"데이터셋 위치: {os.path.join(root, 'VOCdevkit')}")

if __name__ == "__main__":
    main()