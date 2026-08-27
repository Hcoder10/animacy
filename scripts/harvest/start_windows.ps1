# Start the harvest daemon so it outlives the ssh session: a WMI-created process is not part of the
# session's job object (Start-Process children are killed when the OpenSSH session closes).
Remove-Item C:\harvest\data\STOP -ErrorAction SilentlyContinue
$cmd = 'cmd.exe /c "powershell.exe -ExecutionPolicy Bypass -File C:\harvest\hv.ps1 scripts/harvest/daemon.py --n 16 >> C:\harvest\data\logs\daemon.out 2>&1"'
$r = ([wmiclass]"Win32_Process").Create($cmd, "C:\harvest\animacy")
Write-Output ("wmi create rc " + $r.ReturnValue + " pid " + $r.ProcessId)
