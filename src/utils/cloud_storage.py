"""
Artifact Storage & Cloud Synchronization Utility.
Handles local checkpoint serialization and automated Google Drive / Cloud backup.
"""

from typing import Optional, Dict
import os
import shutil
import json
from datetime import datetime


class CloudStorageManager:
    """
    Manages experiment artifact storage, model weights, and cloud synchronization.
    Supports local backup directories and automated Google Drive synchronization.
    """

    def __init__(self, local_artifact_dir: str = "./artifacts", gdrive_target_dir: Optional[str] = None):
        self.local_artifact_dir = local_artifact_dir
        self.gdrive_target_dir = gdrive_target_dir or os.getenv("GDRIVE_BACKUP_PATH", "/content/drive/MyDrive/TomAI_Artifacts")
        os.makedirs(self.local_artifact_dir, exist_ok=True)

    def save_experiment_metadata(self, experiment_id: str, metadata: Dict) -> str:
        """Saves experiment JSON metadata alongside run timestamp."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        record = {
            "experiment_id": experiment_id,
            "timestamp_utc": timestamp,
            "metadata": metadata
        }
        filename = f"{experiment_id}_{timestamp}_meta.json"
        target_path = os.path.join(self.local_artifact_dir, filename)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return target_path

    def sync_to_google_drive(self, checkpoint_path: str, remote_folder_name: str = "checkpoints") -> bool:
        """
        Synchronizes a local checkpoint file to the configured Google Drive target path.
        Detects Google Colab Google Drive mount points seamlessly.
        """
        if not os.path.exists(checkpoint_path):
            print(f"Error: Source checkpoint {checkpoint_path} does not exist.")
            return False

        # If Google Drive is mounted (e.g. on Google Colab or local GDrive desktop client)
        if os.path.exists(os.path.dirname(self.gdrive_target_dir)):
            try:
                dest_dir = os.path.join(self.gdrive_target_dir, remote_folder_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, os.path.basename(checkpoint_path))
                shutil.copy2(checkpoint_path, dest_file)
                print(f"[Cloud Storage] Successfully mirrored checkpoint to Google Drive: {dest_file}")
                return True
            except Exception as e:
                print(f"[Cloud Storage] Failed mirroring to Google Drive: {e}")
                return False
        else:
            print(f"[Cloud Storage] Note: Google Drive destination {self.gdrive_target_dir} not mounted.")
            print(f"[Cloud Storage] Preserved safely in local artifact storage: {checkpoint_path}")
            return False
