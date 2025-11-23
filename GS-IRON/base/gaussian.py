import struct

# ガウスの値（頂いた通り）
gaussian = {
    "x": 0.0270041,
    "y": 0.571603,
    "z": 0.508573,
    "nx": 0.0,
    "ny": 0.0,
    "nz": 0.0,
    "f_dc_0": 3.790736,
    "f_dc_1": 0.610315,
    "f_dc_2": 1.0232366,
    "f_rest_0": 0.102161,
    "f_rest_1": -0.0444992,
    "f_rest_2": -0.00155531,
    "f_rest_3": 0.0192851,
    "f_rest_4": -0.0934443,
    "f_rest_5": -0.0347244,
    "f_rest_6": 0.0230737,
    "f_rest_7": -0.01146,
    "f_rest_8": -0.0111902,
    "f_rest_9": 0.00690077,
    "f_rest_10": -0.00786405,
    "f_rest_11": 0.0208852,
    "f_rest_12": -0.0337452,
    "f_rest_13": 0.00793121,
    "f_rest_14": -0.00659384,
    "f_rest_15": 0.0931693,
    "f_rest_16": -0.0405073,
    "f_rest_17": 0.0159852,
    "f_rest_18": -0.00727745,
    "f_rest_19": -0.0791751,
    "f_rest_20": -0.0150893,
    "f_rest_21": -0.00465871,
    "f_rest_22": 0.0259633,
    "f_rest_23": -0.0108804,
    "f_rest_24": 2.4433e-05,
    "f_rest_25": -0.0225681,
    "f_rest_26": 0.0258566,
    "f_rest_27": -0.0348761,
    "f_rest_28": -0.0254534,
    "f_rest_29": -0.00739053,
    "f_rest_30": 0.0943818,
    "f_rest_31": -0.0197703,
    "f_rest_32": 0.00298108,
    "f_rest_33": -0.00899099,
    "f_rest_34": -0.0713296,
    "f_rest_35": -0.0137983,
    "f_rest_36": 0.000699461,
    "f_rest_37": -0.0243173,
    "f_rest_38": -0.000887608,
    "f_rest_39": 0.00238308,
    "f_rest_40": -0.0124264,
    "f_rest_41": 0.00450711,
    "f_rest_42": -0.0221243,
    "f_rest_43": 0.00137662,
    "f_rest_44": -0.0224955,
    "opacity": 18.81027,
    "scale_0": -0.8,
    "scale_1": -0.8,
    "scale_2": -0.8,
    "rot_0": 0.510566,
    "rot_1": 0.502488,
    "rot_2": 0.00997789,
    "rot_3": -0.140544,
}

# プロパティリスト
properties = [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "nx", "ny", "nz",
    "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
] + [f"f_rest_{i}" for i in range(45)]

# PLYヘッダ
header = f"""ply
format binary_little_endian 1.0
element vertex 1
"""
for prop in properties:
    header += f"property float {prop}\n"
header += "end_header\n"

# バイナリ書き込み
with open("single_gaussian.ply", "wb") as f:
    f.write(header.encode('ascii'))
    for prop in properties:
        f.write(struct.pack('<f', gaussian[prop]))
