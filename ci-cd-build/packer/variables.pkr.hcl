variable "minio_version" {
  type = string
  default = "20221005145827.0.0"
}

variable "cpus" {
  type = number
  default = 2
}

variable "disk_size" {
  description = "Disk size in MB"
  type = number
  default = 51200
}

variable "memory" {
  type = number
  default = 4096
}

variable "headless" {
  type = bool
  default = true
}

variable "http_proxy" {
  type = string
  default = env("http_proxy")
}

variable "https_proxy" {
  type = string
  default = env("https_proxy")
}

variable "no_proxy" {
  type = string
  default = env("no_proxy")
}

variable "hostname" {
  type = string
  default = "minio-ubuntu-20-04"
}

variable "iso_checksum" {
  type = string
  default = "f11bda2f2caed8f420802b59f382c25160b114ccc665dbac9c5046e7fceaced2"
}

variable "iso_checksum_type" {
  type = string
  default = "sha256"
}

variable "iso_name" {
  type = string
  default = "ubuntu-20.04.1-legacy-server-amd64.iso"
}

variable "iso_path" {
  type = string
  default = "iso"
}

variable "iso_url" {
  type = string
  default = "http://cdimage.ubuntu.com/ubuntu-legacy-server/releases/20.04/release/ubuntu-20.04.1-legacy-server-amd64.iso"
}

variable "ssh_fullname" {
  type = string
  default = "vagrant"
}

variable "ssh_username" {
  type = string
  default = "vagrant"
}

variable "ssh_password" {
  type = string
  default = "vagrant"
  sensitive = true
}

variable "minio_service_user" {
  type = string
  default = "minio-user"
}

variable "minio_service_group" {
  type = string
  default = "minio-user"
}

variable "vagrantfile_template" {
  type = string
  default = ""
}

variable "box_version" {
  type = string
  default = "0.1.1"
}

variable "guest_os_type" {
  type = string
  default = "Ubuntu_64"
}

variable "vm_name" {
  type = string
  default = "minio-ubuntu-20-04"
}

variable "vagrant_cloud_token" {
  type = string
  sensitive = true
  default = "abc123"
}

variable "vagrant_cloud_username" {
  type = string
  default = "minio"
}

variable "box_version" {
  type = string
  default = "0.1.0"
}
