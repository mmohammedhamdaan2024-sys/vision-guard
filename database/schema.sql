-- Vision Guard SQLite Database Schema
-- AI-Powered Traffic Violation Detection & Road Safety Monitoring System

CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_id TEXT NOT NULL,
    vehicle_type TEXT NOT NULL,
    plate_number TEXT DEFAULT 'UNREADABLE',
    plate_confidence REAL DEFAULT 0.0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    snapshot_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER,
    tracking_id TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('Critical', 'Warning', 'Review')),
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'Pending Review' CHECK(status IN ('Pending Review', 'Approved', 'Rejected', 'Under Investigation')),
    evidence_path TEXT NOT NULL,
    review_notes TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS challans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    challan_ref TEXT UNIQUE NOT NULL,
    violation_id INTEGER,
    vehicle_plate TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    fine_amount REAL NOT NULL,
    review_status TEXT NOT NULL DEFAULT 'Pending Review' CHECK(review_status IN ('Pending Review', 'Approved', 'Rejected', 'Under Investigation')),
    payment_status TEXT NOT NULL DEFAULT 'Unpaid' CHECK(payment_status IN ('Unpaid', 'Paid', 'Cancelled', 'Disputed')),
    evidence_path TEXT,
    issued_by TEXT DEFAULT 'Vision Guard Automated System',
    issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    mobile_notified INTEGER DEFAULT 0,
    notification_phone TEXT,
    notes TEXT,
    FOREIGN KEY (violation_id) REFERENCES violations(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS review_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    challan_id INTEGER,
    reviewer_name TEXT NOT NULL,
    action TEXT NOT NULL,
    comments TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (challan_id) REFERENCES challans(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_vehicles_tracking ON vehicles(tracking_id);
CREATE INDEX IF NOT EXISTS idx_violations_tracking ON violations(tracking_id);
CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status);
CREATE INDEX IF NOT EXISTS idx_challans_ref ON challans(challan_ref);
CREATE INDEX IF NOT EXISTS idx_challans_plate ON challans(vehicle_plate);
