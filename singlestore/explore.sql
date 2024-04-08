-- Create a new database named 'books'
CREATE DATABASE books;


-- Switch to the 'books' database


USE books;


-- Create a table named 'classic_books' to store information about classic books
CREATE TABLE classic_books
(
   title VARCHAR(255),
   author VARCHAR(255),
   date VARCHAR(255)
);


-- Define a pipeline named 'minio' to load data from an S3 bucket called 'classic-books'
-- The pipeline loads data into the 'classic_books' table


CREATE PIPELINE IF NOT EXISTS minio
AS LOAD DATA S3 'classic-books'
CONFIG '{"region": "us-east-1", "endpoint_url":"http://minio:9000/"}'
CREDENTIALS '{"aws_access_key_id": "minioadmin", "aws_secret_access_key": "minioadmin"}'
INTO TABLE classic_books
FIELDS TERMINATED BY ',';


-- Start the 'minio' pipeline to initiate data loading
START PIPELINE minio;


-- Retrieve and display all records from the 'classic_books' table
SELECT * FROM classic_books;


-- Drop the 'minio' pipeline to stop data loading
DROP PIPELINE minio;


-- Drop the 'classic_books' table to remove it from the database
DROP TABLE classic_books;


-- Drop the 'books' database to remove it entirely
DROP DATABASE books;