# Vision Guard — Data Directory

This directory stores persistent application storage and audit data:

- `visionguard.db`: SQLite 3 database containing vehicle detections, recorded violations, challan registry, and review audit logs.
- `reports/`: Generated PDF audit reports and analytical exports.

## Database Integrity
The SQLite database is initialized automatically by `database/db_manager.py` upon initial application boot if not already present.
