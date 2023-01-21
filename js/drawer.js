H.Drawer = function (viewer) {

  this.nv = viewer.nv;

  this.setupInteraction();

  this.label = 0;
  this.intensity = null;

  this.leftDown = false;
  this.position = null;

  this.tolerance = 30;

  this.magicMode = false;

};


H.Drawer.prototype.getLabelmapPixel = function (x, y, z) {

  let dx = H.D.nv.back.dims[1];
  let dy = H.D.nv.back.dims[2];
  let dz = H.D.nv.back.dims[3];

  return H.D.nv.drawBitmap[x + y * dx + z * dx * dy];

};

H.Drawer.prototype.setLabelmapPixel = function (x, y, z, label) {

  let dx = H.D.nv.back.dims[1];
  let dy = H.D.nv.back.dims[2];
  let dz = H.D.nv.back.dims[3];

  H.D.nv.drawBitmap[x + y * dx + z * dx * dy] = label;

};

H.Drawer.prototype.getVolumePixel = function(x, y, z) {

  return H.D.nv.back.getValue(x,y,z);

};

H.Drawer.prototype.getVolumeDimensions = function() {

  return H.D.nv.back.dims.slice(1);

};


H.Drawer.prototype.setupInteraction = function () {

  // since we are not using the niivue
  // drawing that is builtin, we need
  // to keep track of the mouse position like this
  this.nv.onLocationChange = function (e) {

    this.intensity = e.values[0].value.toFixed(3).replace(/\.?0*$/, "");

    // we just enable drawing for a second to create the array
    if (!this.nv.opts.drawingEnabled) {
      this.nv.setDrawingEnabled(1);
    }
    // but then disable it
    this.nv.setDrawingEnabled(0);

    H.D.position = e['vox'];

  }.bind(this);


  document.getElementById('tolerance').oninput = this.onToleranceChange.bind(this);

  this.nv.canvas.onmousedown = this.onMouseDown.bind(this);
  this.nv.canvas.onmousemove = this.onMouseMove.bind(this);
  this.nv.canvas.onmouseup = this.onMouseUp.bind(this);
  window.onkeypress = this.onKeyPress.bind(this);
  window.onkeydown = this.onKeyDown.bind(this);
  window.onkeyup = this.onKeyUp.bind(this);

};


H.Drawer.prototype.onToleranceChange = function(e) {

  this.tolerance = parseInt(e.target.value, 10);

  document.getElementById('tolerancelabel').innerText = this.tolerance;

};


H.Drawer.prototype.onMouseDown = function (e) {

  H.D.leftDown = true;

  if (e.shiftKey) {

    // activate measuring
    H.V.nv.opts.dragMode = H.V.nv.dragModes.measurement;

  } else {

    H.V.nv.opts.dragMode = H.V.nv.dragModes.slicer3D;

  }

  if (!e.ctrlKey) return;

  this.nv.canvas.style.cursor = 'wait';

  H.D.label += 1;

};


H.Drawer.prototype.onMouseMove = function (e) {

  if (e.ctrlKey) {
    this.nv.canvas.style.cursor = 'crosshair';
  } else {
    this.nv.canvas.style.cursor = 'default';
  }

};


H.Drawer.prototype.onMouseUp = function (e) {

  H.D.leftDown = false;

  if (!e.ctrlKey) return;

  var i = H.D.position[0];
  var j = H.D.position[1];
  var k = H.D.position[2];

  this.intensity = H.D.getVolumePixel(i, j, k);

  H.A.threshold = this.intensity;
  H.A.intensity_max = H.D.nv.back.global_max;
  H.A.threshold_tolerance = H.D.tolerance;
  H.A.label_to_draw = H.D.label;

  H.A.grow(i, j, k);

  H.D.refresh();

  this.nv.canvas.style.cursor = 'default';

};

H.Drawer.prototype.onKeyPress = function(e) {

  if (e.code == 'Space') {
    
    H.V.changeView();

  } else if (e.code == 'KeyZ') {

    H.A.undo();

  } else if (e.code == 'KeyX') {

    H.D.save();

  } else if (e.code == 'KeyA') {

    this.nv.moveCrosshairInVox(-1, 0, 0);

  } else if (e.code == 'KeyD') {

    this.nv.moveCrosshairInVox(1, 0, 0);

  } else if (e.code == 'KeyS') {

    // anterior 
    this.nv.moveCrosshairInVox(0, 1, 0);

  } else if (e.code == 'KeyW') {

    // posterior 
    this.nv.moveCrosshairInVox(0, -1, 0);

  } else if (e.code == 'KeyQ') {

    if (!this.magicMode) {

      // magic mode thanks to Chris 'The Beast' Rorden
      // from: https://niivue.github.io/niivue/features/cactus.html
      H.V.nv.volumes[0].colorMap = "ct_kidneys";
      H.V.nv.volumes[0].cal_min = 130;
      H.V.nv.volumes[0].cal_max = 1000;
      H.V.nv.updateGLVolume();

      this.magicMode = true;

    } else {

      // TODO cleanup to avoid duplication

      H.V.nv.volumes[0].colorMap = "gray";
      H.V.nv.volumes[0].cal_min = 130;
      H.V.nv.volumes[0].cal_max = 1000;
      H.V.nv.updateGLVolume();

      this.magicMode = false;

    }



  }


};


H.Drawer.prototype.onKeyDown = function(e) {

  if (e.key == 'Alt') {

    H.V.nv.drawOpacity = 0.;
    H.V.nv.updateGLVolume();

  }

};


H.Drawer.prototype.onKeyUp = function(e) {

  if (e.key == 'Alt') {

    H.V.nv.drawOpacity = 1.0;
    H.V.nv.updateGLVolume();

  }

};


H.Drawer.prototype.refresh = function() {

  H.D.nv.refreshDrawing();

  var unique_labels = [... new Set(H.D.nv.drawBitmap)].length-1;

  document.getElementById('stats').innerHTML = unique_labels;


};

H.Drawer.prototype.save = function () {

  H.D.nv.saveImage('image.nii.gz', true);

};
