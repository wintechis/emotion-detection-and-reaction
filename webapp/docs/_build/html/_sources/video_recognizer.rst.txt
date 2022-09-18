video\_recognizer module
========================

The video recognizer consists of on function ``gen_frames()`` and a while-loop that generates and returns the videostream with overlays. A function can only return one object, but we need the videostream and the predictions.
To solve this, the predictions are saved to a JSON file and constantly updated during runtime.

.. automodule:: video_recognizer
   :members:
   :undoc-members:
   :show-inheritance:
