CREATE TABLE IF NOT EXISTS Products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    specs TEXT
);

INSERT INTO Products (name, category, specs)
VALUES
    ('Prestige Pressure cooker','cook','lightweight 1.5L'),
    ('HP Laptop', 'electronics','8GB RAM 3.5GHz 2kg'),
    ('Purpleant Sit stand desk', 'furniture', 'height adjustible white manual');
