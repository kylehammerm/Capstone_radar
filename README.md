This version will save tensor file and label them. You'll be prompted a label once you hit Start record and the recording starts immediately you click ''Ok''. Start and Stop are the same button so you dont have to worry or stress about not hitting the right one. 

Correct way to run live -> run "realAWR2944.py" -> Get label, about 15+ samples of each -> run "cnn_autov1.py --train" to train on your samples(I'll also post my samples as alternatives) -> run "cnn_autov1.py --test" to test and see result of training(this is optional). For live demo, run "live_demo.py" after you edited the script.

realAWR2944.py:
You don't need to record your own samples since I can provide mine. You can simply add to it by performing the right gestures if you want. Note that, there will be noise, any movement from other body parts, including head nudge, moving mouse and cursor to end recording, "BREATHING(JK)" will be picked up by the radar. So, it's best to seek someone else to assist you in recording labels.
Labels should be saved in /cnn_tensors/{label}/"{label}_{timestamp}.npy" You'll make use of this path when running cnn this should already be implemented, so no need to change.

cnn_autov1.py:
You must use params --train --epochs --test --visualize --debug
To train, you can run this is cmd "python cnn_autov1.py --train --epochs 40" Default epochs is set to 100 though you dont need that much and it wastes your time. This trains on 80% of your data, and uses the rest of the 20% to test its capabilities.

To test, which is optional, run this in cmd: "python cnn_autov1.py --test", You should see prediction charts, and accuracy figures. You can also make use to the logs in the console.

For a live demo, you have to fix the bug in the script first. No clean exit button in the GUI to properly exit the radar and DCA. Streamming still continues even after keyboard interrupt or closing the GUI window.
Remove excessive logging of predictions since the GUI window is enough to show what it's doing. To run this, make sure you have the .pth cnn model maybe (this should be in this path) /checkpoint/.pth, THEN run it with play or 'python live_demo.py'. 

