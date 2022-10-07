#!/bin/bash -eu

SSH_USER=${SSH_USERNAME:-vagrant}

echo "==> Cleaning up tmp"
rm -rf /tmp/*

echo "==> Installed packages"
dpkg --get-selections | grep -v deinstall

# Remove Bash history
unset HISTFILE
rm -f /root/.bash_history
rm -f /home/${SSH_USER}/.bash_history

echo "==> Clearing machine-id"
truncate --size=0 /etc/machine-id

echo "==> Clearing log files"
find /var/log -type f -exec truncate --size=0 {} \;

echo '==> Clear out swap and disable until reboot'
set +e
swapuuid=$(blkid -o value -l -s UUID -t TYPE=swap)
case "$?" in
    2|0) ;;
    *) exit 1 ;;
esac
set -e
if [ "x$swapuuid" != "x" ]; then
    # Whiteout the swap partition to reduce box size
    # Swap is disabled till reboot
    swappart=$(readlink -f /dev/disk/by-uuid/${swapuuid})
    swapoff $swappart
    dd if=/dev/zero of=$swappart bs=1M || echo "dd exit code $? is suppressed"
    mkswap -U $swapuuid $swappart
fi

# Zero out the free space to save space in the final image
if [ -d /boot/efi ]; then
    dd if=/dev/zero of=/boot/efi/EMPTY bs=1M || echo "dd exit code $? is suppressed"
    rm -f /boot/efi/EMPTY
fi
dd if=/dev/zero of=/EMPTY bs=1M || echo "dd exit code $? is suppressed"
rm -f /EMPTY
sync

echo "==> Disk usage after cleanup"
df -h