from langflow.custom import Component
from langflow.inputs import DataInput, SecretStrInput, StrInput, DropdownInput
from langflow.io import Output
from langflow.schema import Data
from typing import Dict, Any, List, Tuple
from twelvelabs import TwelveLabs
import time
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor

class PegasusIndexVideo(Component):
    """Indexes videos using Twelve Labs Pegasus API and adds the video ID to metadata."""

    display_name = "Twelve Labs Pegasus Index Video"
    description = "Index videos using Twelve Labs and add the video_id to metadata."
    icon = "TwelveLabs"
    name = "TwelveLabsPegasusIndexVideo"
    documentation = "https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md"

    inputs = [
        DataInput(
            name="videodata", 
            display_name="Video Data", 
            info="Video Data objects (from VideoFile or SplitVideo)",
            is_list=True,
            required=True
        ),
        SecretStrInput(
            name="api_key",
            display_name="Twelve Labs API Key",
            info="Enter your Twelve Labs API Key.",
            required=True
        ),
        DropdownInput(
            name="model_name",
            display_name="Model",
            info="Pegasus model to use for indexing",
            options=["pegasus1.2"],
            value="pegasus1.2",
            advanced=False,
        ),
        StrInput(
            name="index_name",
            display_name="Index Name",
            info="Name of the index to use. If the index doesn't exist, it will be created.",
            required=False
        ),
        StrInput(
            name="index_id",
            display_name="Index ID",
            info="ID of an existing index to use. If provided, index_name will be ignored.",
            required=False
        ),
    ]

    outputs = [
        Output(
            display_name="Indexed Data",
            name="indexed_data",
            method="index_videos",
            output_types=["Data"],
            is_list=True
        ),
    ]

    def _get_or_create_index(self, client: TwelveLabs) -> Tuple[str, str]:
        """Get existing index or create new one. Returns (index_id, index_name)"""
        
        # First check if index_id is provided and valid
        if hasattr(self, 'index_id') and self.index_id:
            try:
                index = client.index.retrieve(id=self.index_id)
                return self.index_id, index.name
            except Exception as e:
                if not hasattr(self, 'index_name') or not self.index_name:
                    raise ValueError("Invalid index ID provided and no index name specified for fallback.")

        # If index_name is provided, try to find it
        if hasattr(self, 'index_name') and self.index_name:
            try:
                # List all indexes and find by name
                indexes = client.index.list()
                for idx in indexes:
                    if idx.name == self.index_name:
                        return idx.id, idx.name
                
                # If we get here, index wasn't found - create it
                index = client.index.create(
                    name=self.index_name,
                    models=[
                        {
                            "name": self.model_name if hasattr(self, 'model_name') else "pegasus1.2",
                            "options": ["visual", "audio"]
                        }
                    ]
                )
                return index.id, index.name
            except Exception as e:
                raise

        # If we get here, neither index_id nor index_name was provided
        raise ValueError("Either index_name or index_id must be provided")

    def on_task_update(self, task, video_path):
        """Callback for task status updates"""
        status_msg = f"Indexing {os.path.basename(video_path)}... Status: {task.status}"
        self.status = status_msg

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        reraise=True
    )
    def _check_task_status(
        self, 
        client: TwelveLabs, 
        task_id: str, 
        video_path: str,
    ) -> Dict[str, Any]:
        """Check task status once"""
        task = client.task.retrieve(id=task_id)
        self.on_task_update(task, video_path)
        return task

    def _wait_for_task_completion(
        self, 
        client: TwelveLabs, 
        task_id: str, 
        video_path: str,
        max_retries: int = 120,
        sleep_time: int = 10
    ) -> Dict[str, Any]:
        """Wait for task completion with timeout and improved error handling"""
        retries = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while retries < max_retries:
            try:
                self.status = f"Checking task status for {os.path.basename(video_path)} (attempt {retries + 1})"
                task = self._check_task_status(client, task_id, video_path)

                if task.status == "ready":
                    self.status = f"Indexing for {os.path.basename(video_path)} completed successfully!"
                    return task
                elif task.status == "failed":
                    error_msg = f"Task failed for {os.path.basename(video_path)}: {getattr(task, 'error', 'Unknown error')}"
                    self.status = error_msg
                    raise Exception(error_msg)
                elif task.status == "error":
                    error_msg = f"Task encountered an error for {os.path.basename(video_path)}: {getattr(task, 'error', 'Unknown error')}"
                    self.status = error_msg
                    raise Exception(error_msg)
                
                time.sleep(sleep_time)
                retries += 1
                elapsed_time = retries * sleep_time
                self.status = f"Indexing {os.path.basename(video_path)}... {elapsed_time}s elapsed"
                
            except Exception as e:
                consecutive_errors += 1
                error_msg = f"Error checking task status for {os.path.basename(video_path)}: {str(e)}"
                self.status = error_msg
                
                if consecutive_errors >= max_consecutive_errors:
                    raise Exception(f"Too many consecutive errors checking task status for {os.path.basename(video_path)}: {error_msg}")
                
                time.sleep(sleep_time * (2 ** consecutive_errors))
                continue
        
        timeout_msg = f"Timeout waiting for indexing of {os.path.basename(video_path)} after {max_retries * sleep_time} seconds"
        self.status = timeout_msg
        raise TimeoutError(timeout_msg)

    def _upload_video(self, client: TwelveLabs, video_path: str, index_id: str) -> str:
        """Upload a single video and return its task ID"""
        with open(video_path, 'rb') as video_file:
            self.status = f"Uploading {os.path.basename(video_path)} to index {index_id}..."
            task = client.task.create(
                index_id=index_id,
                file=video_file
            )
            task_id = task.id
            self.status = f"Upload complete for {os.path.basename(video_path)}. Task ID: {task_id}"
            return task_id

    def index_videos(self) -> List[Data]:
        """Indexes each video and adds the video_id to its metadata."""
        if not self.videodata:
            self.status = "No video data provided."
            return []
        
        if not self.api_key:
            raise ValueError("Twelve Labs API Key is required.")

        if not (hasattr(self, 'index_name') and self.index_name) and not (hasattr(self, 'index_id') and self.index_id):
            raise ValueError("Either index_name or index_id must be provided")

        client = TwelveLabs(api_key=self.api_key)
        indexed_data_list = []
        
        # Get or create the index
        try:
            index_id, index_name = self._get_or_create_index(client)
            self.status = f"Using index: {index_name} (ID: {index_id})"
        except Exception as e:
            self.status = f"Failed to get/create Twelve Labs index: {str(e)}"
            raise

        # First, validate all videos and create a list of valid ones
        valid_videos: List[Tuple[Data, str]] = []
        for video_data_item in self.videodata:
            if not isinstance(video_data_item, Data):
                self.status = f"Skipping invalid data item: {video_data_item}"
                continue

            video_info = video_data_item.data
            if not isinstance(video_info, dict):
                self.status = f"Skipping item with invalid data structure: {video_info}"
                continue

            video_path = video_info.get('text')
            if not video_path or not isinstance(video_path, str):
                self.status = f"Skipping item with missing or invalid video path: {video_info}"
                continue

            if not os.path.exists(video_path):
                self.status = f"Video file not found, skipping: {video_path}"
                continue
            
            valid_videos.append((video_data_item, video_path))

        if not valid_videos:
            self.status = "No valid videos to process."
            return []

        # Upload all videos first and collect their task IDs
        upload_tasks: List[Tuple[Data, str, str]] = []  # (data_item, video_path, task_id)
        for data_item, video_path in valid_videos:
            try:
                task_id = self._upload_video(client, video_path, index_id)
                upload_tasks.append((data_item, video_path, task_id))
            except Exception as e:
                self.status = f"Failed to upload {video_path}: {str(e)}"
                continue

        # Now check all tasks in parallel using a thread pool
        with ThreadPoolExecutor(max_workers=min(10, len(upload_tasks))) as executor:
            futures = []
            for data_item, video_path, task_id in upload_tasks:
                future = executor.submit(
                    self._wait_for_task_completion,
                    client,
                    task_id,
                    video_path
                )
                futures.append((data_item, video_path, future))

            # Process results as they complete
            for data_item, video_path, future in futures:
                try:
                    completed_task = future.result()
                    if completed_task.status == "ready":
                        video_id = completed_task.video_id
                        self.status = f"Video {os.path.basename(video_path)} indexed successfully. Video ID: {video_id}"
                        
                        # Add video_id to the metadata
                        video_info = data_item.data
                        if 'metadata' not in video_info:
                            video_info['metadata'] = {}
                        elif not isinstance(video_info['metadata'], dict):
                            self.status = f"Warning: Overwriting non-dict metadata for {video_path}"
                            video_info['metadata'] = {}

                        video_info['metadata'].update({
                            'video_id': video_id,
                            'index_id': index_id,
                            'index_name': index_name
                        })
                        
                        updated_data_item = Data(data=video_info)
                        indexed_data_list.append(updated_data_item)
                except Exception as e:
                    self.status = f"Failed to process {video_path}: {str(e)}"

        if not indexed_data_list:
            self.status = "No videos were successfully indexed."
        else:
            self.status = f"Finished indexing {len(indexed_data_list)}/{len(self.videodata)} videos."
        
        return indexed_data_list
