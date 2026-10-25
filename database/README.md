# VisionGuard — Database Architecture

VisionGuard uses a local SQLite 3 database (`data/visionguard.db`) with relational integrity, indices, and audit logging.

## Schema: `violation_events`
- `id`: Primary key (autoincrement)
- `event_id`: Unique identifier (e.g. `EV-202610-84920`)
- `timestamp`: Event creation time
- `violation_category`: Categorical safety violation
- `tracking_id`: Vehicle/Person tracking identity
- `vehicle_type`: Entity class (car, motorcycle, bus, truck, bicycle, person)
- `confidence_score`: Detection confidence (0.0 to 1.0)
- `evidence_path`: Local path to watermarked high-resolution frame crop
- `camera_source`: Feed origin ('Live Camera' / 'Uploaded Video')
- `review_status`: Review state ('Pending Review', 'Confirmed', 'Rejected', 'Inconclusive')
- `reviewer_notes`: Officer adjudication comments
