"""Generated gRPC stubs for nvidia_ace.services.a2f.v1.A2FService.

Source protos live alongside this file (copied from NVIDIA ACE's
``audio_2_face_microservice/1.2/proto/protobuf_files/``) and have been
renamed from dotted names (``nvidia_ace.a2f.v1.proto``) to underscored
names (``nvidia_ace_a2f_v1.proto``) so the generated Python modules are
importable. Internal ``import "..."`` statements were rewritten to match.

If you ever need to regenerate the stubs, from ``backend/``:

    uv run python -m grpc_tools.protoc \\
        -I pipeline/a2f/proto \\
        --python_out=pipeline/a2f/proto \\
        --grpc_python_out=pipeline/a2f/proto \\
        pipeline/a2f/proto/nvidia_ace_services_a2f_v1.proto \\
        pipeline/a2f/proto/nvidia_ace_a2f_v1.proto \\
        pipeline/a2f/proto/nvidia_ace_audio_v1.proto \\
        pipeline/a2f/proto/nvidia_ace_animation_id_v1.proto \\
        pipeline/a2f/proto/nvidia_ace_status_v1.proto \\
        pipeline/a2f/proto/nvidia_ace_emotion_with_timecode_v1.proto

Then patch the generated ``_pb2.py`` / ``_pb2_grpc.py`` files so their
sibling-module imports are relative (protoc emits absolute imports that
don't work from inside a package):

    sed -i -E 's|^import (nvidia_ace_[a-z0-9_]+_pb2)( as .*)?$|from . import \\1\\2|' \\
        pipeline/a2f/proto/*_pb2.py pipeline/a2f/proto/*_pb2_grpc.py
"""
