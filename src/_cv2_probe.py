import cv2
print("cv2", cv2.__version__)
print("KCF_create", hasattr(cv2, "TrackerKCF_create"))
print("KCF_legacy", hasattr(cv2, "TrackerKCF"))
print("MIL_create", hasattr(cv2, "TrackerMIL_create"))
print("CSRT_create", hasattr(cv2, "TrackerCSRT_create"))
print("legacy", hasattr(cv2, "legacy"))
