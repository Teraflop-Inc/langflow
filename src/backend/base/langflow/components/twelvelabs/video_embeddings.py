from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import SecretStrInput, IntInput, DropdownInput
from twelvelabs import TwelveLabs
import time
from typing import List, Dict, Any, Optional, Union, Sequence, cast

class TwelveLabsVideoEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "Marengo-retrieval-2.7") -> None:
        self.client = TwelveLabs(api_key=api_key)
        self.model_name = model_name
        
    def _wait_for_task_completion(self, task_id: str) -> Any:
        while True:
            result = self.client.embed.task.retrieve(id=task_id)
            if result.status == "ready":
                return result
            time.sleep(5)
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            video_path = text.page_content if hasattr(text, 'page_content') else str(text)
            result = self.embed_video(video_path)
            
            # First try to use video embedding, then fall back to clip embedding if available
            if result['video_embedding'] is not None:
                embeddings.append(cast(List[float], result['video_embedding']))
            elif result['clip_embeddings'] and len(result['clip_embeddings']) > 0:
                embeddings.append(cast(List[float], result['clip_embeddings'][0]))
            else:
                # If neither is available, raise an error
                raise ValueError("No embeddings were generated for the video")
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        video_path = text.page_content if hasattr(text, 'page_content') else str(text)
        result = self.embed_video(video_path)
        
        # First try to use video embedding, then fall back to clip embedding if available
        if result['video_embedding'] is not None:
            return cast(List[float], result['video_embedding'])
        elif result['clip_embeddings'] and len(result['clip_embeddings']) > 0:
            return cast(List[float], result['clip_embeddings'][0])
        else:
            # If neither is available, raise an error
            raise ValueError("No embeddings were generated for the video")

    def embed_video(self, video_path: str) -> Dict[str, Union[List[float], List[List[float]]]]:
        with open(video_path, 'rb') as video_file:
            task = self.client.embed.task.create(
                model_name=self.model_name,
                video_file=video_file,
                video_embedding_scopes=["video","clip"],
            )
        
        result = self._wait_for_task_completion(task.id)
        
        video_embedding: Dict[str, Union[List[float], List[List[float]]]] = {
            'video_embedding': [],  # Initialize as empty list instead of None
            'clip_embeddings': []
        }
        
        if hasattr(result.video_embedding, 'segments') and result.video_embedding.segments:
            for seg in result.video_embedding.segments:
                # Check for embeddings_float attribute (this is the correct attribute name)
                if hasattr(seg, 'embeddings_float'):
                    if seg.embedding_scope == "video":
                        # Convert to list of floats
                        video_embedding['video_embedding'] = [float(x) for x in seg.embeddings_float]
        
        return video_embedding

class TwelveLabsVideoEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "Twelve Labs Video Embeddings"
    description = "Generate embeddings from videos using Twelve Labs video embedding models."
    name = "TwelveLabsVideoEmbeddings"
    icon = "TwelveLabs"
    documentation = "https://github.com/twelvelabs-io/twelvelabs-developer-experience/blob/main/integrations/Langflow/TWELVE_LABS_COMPONENTS_README.md"
    inputs = [
        SecretStrInput(
            name="api_key", 
            display_name="API Key", 
            required=True
        ),
        DropdownInput(
            name="model_name",
            display_name="Model",
            advanced=False,
            options=["Marengo-retrieval-2.7"],
            value="Marengo-retrieval-2.7",
        ),
        IntInput(
            name="request_timeout",
            display_name="Request Timeout",
            advanced=True
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        return TwelveLabsVideoEmbeddings(
            api_key=self.api_key,
            model_name=self.model_name
        )
