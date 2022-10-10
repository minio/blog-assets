source "virtualbox-iso" "minio-vbox" {

  vm_name = var.vm_name

  cpus      = var.cpus
  disk_size = var.disk_size
  memory    = var.memory

  hard_drive_interface = "sata"
  headless             = var.headless

  iso_checksum = "${var.iso_checksum_type}:${var.iso_checksum}"
  iso_urls     = [
        "${var.iso_path}/${var.iso_name}",
        var.iso_url
      ]

  guest_os_type           = var.guest_os_type
  guest_additions_path    = "VBoxGuestAdditions_{{.Version}}.iso"
  virtualbox_version_file = ".vbox_version"

  http_directory = "http"
  boot_wait      = "5s"
  boot_command = [
        "<esc><esc><enter><wait>",
        "/install/vmlinuz noapic",
        " initrd=/install/initrd.gz",
        " auto=true",
        " priority=critical",
        " hostname=${var.hostname}",
        " passwd/user-fullname=${var.ssh_fullname}",
        " passwd/username=${var.ssh_username}",
        " passwd/user-password=${var.ssh_password}",
        " passwd/user-password-again=${var.ssh_password}",
        " preseed/url=http://{{.HTTPIP}}:{{.HTTPPort}}/preseed.cfg",
        " -- <enter>"
      ]

  ssh_password = "${var.ssh_password}"
  ssh_username = "${var.ssh_username}"
  ssh_timeout  = "4h"

  vboxmanage = [
        ["modifyvm", "{{.Name}}", "--audio", "none"],
        ["modifyvm", "{{.Name}}", "--usb", "off"],
        ["modifyvm", "{{.Name}}", "--vram", "12"],
        ["modifyvm", "{{.Name}}", "--vrde", "off"],
        ["modifyvm", "{{.Name}}", "--nictype1", "virtio"],
        ["modifyvm", "{{.Name}}", "--memory", var.memory],
        ["modifyvm", "{{.Name}}", "--cpus", var.cpus]
      ]

  shutdown_command = "echo '${var.ssh_password}'|sudo -S shutdown -P now"
  output_directory = "output-${var.vm_name}-virtualbox-iso"
  
}

build {
  sources = ["sources.virtualbox-iso.minio-vbox"]

  provisioner "shell" {

    environment_vars = [
        "DEBIAN_FRONTEND=noninteractive",
        "SSH_USERNAME=${var.ssh_username}",
        "SSH_PASSWORD=${var.ssh_password}",
        "MINIO_VERSION=${var.minio_version}",
        "MINIO_SERVICE_USER=${var.minio_service_user}",
        "MINIO_SERVICE_GROUP=${var.minio_service_group}",
        "http_proxy=${var.http_proxy}",
        "https_proxy=${var.https_proxy}",
        "no_proxy=${var.no_proxy}"
      ]

    scripts = [
        "scripts/setup.sh",
        "scripts/vagrant.sh",
        "scripts/minio.sh",
        "scripts/cleanup.sh"
      ]

    execute_command   = "echo '${var.ssh_password}'|{{.Vars}} sudo -E -S bash '{{.Path}}'"
    expect_disconnect = true
  }

  post-processors {  
  
    post-processor "vagrant" {
      output = "box/{{.Provider}}/${var.vm_name}-${var.box_version}.box"
      keep_input_artifact  = true
      provider_override    = "virtualbox"
      vagrantfile_template = var.vagrantfile_template
    }

    post-processor "vagrant-cloud" {
      access_token = "${var.vagrant_cloud_token}"
      box_tag      = "${var.vagrant_cloud_username}/${var.vm_name}"
      version      = var.box_version
    }

    post-processor "shell-local" {
      inline = ["rm -rf output-${var.vm_name}-virtualbox-iso"]
    }

  }

}