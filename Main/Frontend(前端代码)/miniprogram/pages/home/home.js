// pages/home/home.js
const MAX_WAIT_TIME = 20000
Page({
  data: {
    imgUrl: '',
    // 选择栏数据（可自定义）
    typeArray: ['启用', '关闭'],
    typeIndex: 1,
    levelArray: ['启用', '关闭'],
    levelIndex: 1,
    statusArray: ['正常', '异常', '待处理'],
    statusIndex: 1,
    // 位置信息
    locationText: "未获取", // 默认占位
    latitude: null,
    longitude: null,
    // 🔥 新增：其他信息相关状态
    extraArray: ['是', '否'],
    extraIndex: 1, // 默认值为 1（对应“否”），即默认隐藏文本框
    extraInputSoil: '', // 字段1：存放土壤信息（原 extraInput）
    extraInputPlant: '' // 字段2：🔥 新增：存放种植情况
  },

  // ✅ 兼容开发者工具的分支判断版本
  chooseImage() {
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album', 'camera'],
      success: (res) => {
        const tempFilePath = res.tempFilePaths[0];

        // 获取系统信息
        const sysInfo = wx.getSystemInfoSync();

        // 判断是否为开发者工具 (devtools)
        if (sysInfo.platform === 'devtools') {
          console.log('当前为开发者模式，跳过裁剪直接上传');
          this.setData({
            imgUrl: tempFilePath
          });
          // 如果你后面有上传逻辑（如 uploadFile），可以在这里直接调用
          return;
        }

        // 手机真机环境下，正常调用裁剪
        wx.editImage({
          src: tempFilePath,
          success: (editRes) => {
            this.setData({
              imgUrl: editRes.tempFilePath
            });
          },
          fail: (err) => {
            console.log('用户取消裁剪或裁剪失败', err);
            // 策略：如果裁剪接口调用失败（部分老机型），也可以保底直接显示原图
            this.setData({
              imgUrl: tempFilePath
            });
          }
        });
      },
      fail: (err) => {
        console.log('用户取消选图', err);
      }
    });
  },

  // 选择栏事件（完全保留）
  bindTypeChange(e) {
    this.setData({
      typeIndex: e.detail.value
    })
  },
  bindLevelChange(e) {
    this.setData({
      levelIndex: e.detail.value
    })
  },
  bindStatusChange(e) {
    this.setData({
      statusIndex: e.detail.value
    })
  },
  // 监听“其他信息”Picker 切换
  bindExtraChange(e) {
    const val = parseInt(e.detail.value, 10);
    this.setData({
      extraIndex: val
    });
    // 如果用户切回“否”，顺便把两个输入框内容全清空，防止残留数据被意外提交
    if (val === 1) {
      this.setData({
        extraInputSoil: '',
        extraInputPlant: ''
      });
    }
  },

  // 监听文本框 1 的实时输入（环境补充信息）
  bindSoilInput(e) {
    this.setData({
      extraInputSoil: e.detail.value
    });
  },

  // 🔥 新增：监听文本框 2 的实时输入（施肥灌溉情况）
  bindPlantInput(e) {
    this.setData({
      extraInputPlant: e.detail.value
    });
  },
  resetPage() {
    this.setData({
      imgUrl: '', // 清空图片
      typeIndex: 1, // 类型选择器重置
      levelIndex: 1, // 等级选择器重置
      statusIndex: 1, // 状态选择器重置
      extraIndex: 1, // 重置回“否”
      extraInputSoil: '', // 清空文本内容
      extraInputPlant: '' // 清空文本内容
    })

    wx.showToast({
      title: '已重置',
      icon: 'success'
    })
  },
  getLocation() {
    const that = this;

    // 先判断权限
    wx.getSetting({
      success(res) {
        // 已授权
        if (res.authSetting['scope.userLocation']) {
          wx.getLocation({
            type: 'gcj02',
            success(res) {
              const lat = res.latitude.toFixed(6)
              const lng = res.longitude.toFixed(6)
              that.setData({
                latitude: lat,
                longitude: lng,
                locationText: lat + ", " + lng
              })
              wx.showToast({
                title: '定位成功'
              })
            },
            fail() {
              wx.showModal({
                title: '定位失败',
                content: '请在手机设置开启定位服务',
                showCancel: false
              })
            }
          })
        }
        // 未授权 → 弹授权
        else {
          wx.authorize({
            scope: 'scope.userLocation',
            success() {
              that.getLocation() // 授权成功重新获取
            },
            fail() {
              wx.showModal({
                title: '无法获取位置',
                content: '请在设置中打开定位权限',
                showCancel: false
              })
            }
          })
        }
      }
    })
  },

  /**
   * 完整的提交逻辑：包含身份校验、云端/本地分支
   */
  async submitData() {
    console.log('--- [DEBUG] 提交按钮被点击 ---');
    const {
      imgUrl
    } = this.data;

    if (!imgUrl) {
      wx.showToast({
        title: '请先上传图片',
        icon: 'none'
      });
      return;
    }

    const userInfo = getApp().globalData.userInfo || wx.getStorageSync('userInfo');
    const isGuest = !userInfo || !userInfo.openid;
    console.log('当前登录态:', isGuest ? '游客模式' : '正式用户');

    // 统一调用 saveToCloud，内部会根据返回的 _id 进行本地持久化
    try {
      await this.saveToCloud(isGuest);
    } catch (err) {
      console.error('提交失败:', err);
    }
  },

  async saveToCloud(isGuest) {
    const {
      imgUrl,
      latitude,
      longitude
    } = this.data;
    wx.showModal({
      title: '诊断启动',
      content: 'AI 诊断已开始，由于模型复杂，可能需要十余秒，请耐心等待。',
      showCancel: false, // 隐藏取消按钮，强制用户看提示
      confirmText: '好的'
    });
    wx.showLoading({
      title: 'AI 诊断中...',
      mask: true
    });

    try {
      // 1. 仅上传图片到云存储
      const cloudPath = `records/${Date.now()}.jpg`;
      const uploadRes = await wx.cloud.uploadFile({
        cloudPath,
        filePath: imgUrl,
      });
      const fileID = uploadRes.fileID;

      // 2. 直接启动 AI 分析，不预先写数据库
      // 注意：此时我们没有 serverId，传 null
      this.triggerAIAnalysis(null, fileID, latitude, longitude, isGuest);

    } catch (err) {
      wx.hideLoading();
      wx.showModal({
        title: '处理失败',
        content: err.message,
        showCancel: false
      });
    }
  },

  // 修改后的 triggerAIAnalysis：拿到结果后“一键保存”
  // home.js


  async triggerAIAnalysis(dummyId, cloudFileID, lat, lng, isGuest) {
    const {
      typeArray,
      typeIndex,
      statusArray,
      statusIndex,
      levelIndex,
      extraIndex, // 👈 新增：解构 extraIndex
      extraInputSoil,
      extraInputPlant
      // 👈 新增：解构 extraInput
    } = this.data;
    // 2. 将索引转换为布尔值：如果是 0 (启用) 则为 true，否则为 false
    const useDipBoolean = (typeIndex === 0);
    const useSmart = (levelIndex === 0);
    const soilinfo = extraInputSoil;
    const plantinfo = extraInputPlant;

    let isFinished = false; // 标记位：记录是否已经正常完成

    // 1. 开启计时器：超时处理
    const timeoutId = setTimeout(() => {
      if (!isFinished) {
        isFinished = true; // 锁定状态，防止后续回调继续执行
        wx.hideLoading();
        wx.showModal({
          title: '诊断超时',
          content: '由于网络较慢或 AI 响应延迟，请稍后在记录中查看或重试。',
          showCancel: false,
          confirmText: '我知道了'
        });
        // 注意：这里无法直接“杀死”已经发出的 wx.request，
        // 但通过 isFinished 标记位，我们可以拦截掉后面成功后的 UI 操作
      }
    }, MAX_WAIT_TIME);

    try {
      const tempUrlRes = await wx.cloud.getTempFileURL({
        fileList: [cloudFileID]
      });
      const httpsImageUrl = tempUrlRes.fileList[0].tempFileURL;
      const typeText = this.data.typeArray[this.data.typeIndex]; // 图像增强：启用/关闭
      const levelText = this.data.levelArray[this.data.levelIndex]; // 启智模式：启用/关闭
      const statusText = this.data.statusArray[this.data.statusIndex]; // 状态：正常/异常



      // 现场用你的云托管环境 ID 实例化一个 c1 对象
      const c1 = new wx.cloud.Cloud({
        resourceEnv: 'corn-diseases-d8g769zyf5d52a368' // 👈 你的云托管环境 ID
      });
      // 必须先调用 init
      await c1.init();
      // 2. 发起请求
      const requestTask = c1.callContainer({
        path: '/api/process',
        method: 'POST',
        header: {
          'X-WX-SERVICE': 'corn-disease'
        },
        data: {
          image_url: httpsImageUrl,
          latitude: lat,
          longitude: lng,
          use_dip: useDipBoolean, // 👈 传入转换后的布尔值
          smart_mode: useSmart,
          soil_type: soilinfo,
          planting_density: plantinfo
        },
        success: async (res) => {
          // 关键：如果已经超时，则不执行任何操作
          if (isFinished) return;

          if (res.data && res.data.code === 0) {
            const ai = res.data.data;
            const addressInfo = ai.location || {};
            // 💡 【修正点】：直接使用方法顶部已经解构好的局部变量 soilinfo 和 plantinfo
            const finalSoilRemark = this.data.extraIndex === 0 ? (soilinfo || "") : "";
            const finalPlantRemark = this.data.extraIndex === 0 ? (plantinfo || "") : "";


            const finalRecordData = {
              image_cloud_id: cloudFileID,
              is_guest: isGuest,
              result: [{
                diseaseName: typeText,
                level: levelText
              }],
              health_score: ai.health_score,
              suggestion: ai.suggestion,
              location: {
                lat: parseFloat(lat),
                lng: parseFloat(lng),
                province: addressInfo.province || "",
                city: addressInfo.city || "",
                district: addressInfo.district || ""
              },
              // 📦 🔥 核心修改：将两个输入框的内容归纳进 env_data 传给云函数
              env_data: {
                soil_remark: finalSoilRemark,
                plant_remark: finalPlantRemark // 施肥与灌溉情况
              },
              status: '分析完成'
            };

            // 写入数据库
            const saveRes = await wx.cloud.callFunction({
              name: 'saveRecord',
              data: {
                recordData: finalRecordData
              }
            });

            // 3. 正常完成逻辑
            clearTimeout(timeoutId); // 务必清除定时器
            isFinished = true;
            wx.hideLoading();

            if (saveRes.result.code === 0) {
              wx.showModal({
                title: '诊断完成',
                content: '请前往记录中查看检测结果',
                showCancel: false,
                confirmText: '我知道了'
              });
              this.setData({
                imgUrl: '',
                typeIndex: 1,
                levelIndex: 1,
                statusIndex: 1,
                extraIndex: 1, // 👈 建议：重置时一并恢复“其他信息”的选择
                extraInputSoil: '', // 👈 建议：重置时一并清空输入框
                extraInputPlant: ''
              });
            }
          }
        },
        fail: (err) => {
          if (isFinished) return;
          clearTimeout(timeoutId);
          isFinished = true;
          wx.hideLoading();
          wx.showModal({
            title: '请求失败',
            content: '连接后端服务器失败，请检查网络。',
            showCancel: false
          });
        }
      });

    } catch (e) {
      if (!isFinished) {
        clearTimeout(timeoutId);
        wx.hideLoading();
        console.error(e);
      }
    }
  },

  // 辅助方法：格式化时间对齐你的图一格式
  formatDate(date) {
    const Y = date.getFullYear();
    const M = (date.getMonth() + 1).toString().padStart(2, '0');
    const D = date.getDate().toString().padStart(2, '0');
    const h = date.getHours().toString().padStart(2, '0');
    const m = date.getMinutes().toString().padStart(2, '0');
    const s = date.getSeconds().toString().padStart(2, '0');
    return `${Y}-${M}-${D} ${h}:${m}:${s}`;
  }

})