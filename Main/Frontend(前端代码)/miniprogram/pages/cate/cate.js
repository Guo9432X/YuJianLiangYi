Page({
  data: {
    currentTab: 1,
    records: [], // 本地记录
    cloudRecords: [], // 新增：云端记录
    isLoading: false, // 新增：加载状态
    showModal: false,
    currentRecord: {}
  },

  onShow() {
    this.loadRecords();
    // 如果当前就在云端 Tab，则刷新云端数据
    if (this.data.currentTab === 1) {
      this.loadCloudRecords();
    }
  },

  loadRecords() {
    try {
      const records = wx.getStorageSync('upload_records') || []
      this.setData({
        records
      })
    } catch (e) {
      this.setData({
        records: []
      })
    }
  },

  // 🔥 打开弹窗
  // 🔥 打开弹窗并获取详情
  // cate.js
  async openModal(e) {
    const record = e.currentTarget.dataset.record;
    this.setData({
      currentRecord: record,
      showModal: true,
      isDetailLoading: true
    });

    if (record._id) {
      try {
        const res = await wx.cloud.callFunction({
          name: 'getRecordDetail',
          data: {
            recordId: record._id
          }
        });

        if (res.result && res.result.code === 0) {
          const detail = res.result.data;
          const sug = detail.suggestion;

          let formattedText = '';

          if (sug && typeof sug === 'object') {
            // 1. 提取当前分析 (current_analysis)
            if (sug.current_analysis) {
              formattedText += `【专家分析】\n${sug.current_analysis}\n\n`;
            }

            // 2. 提取农业防治 (agricultural_control)
            if (sug.control_measures && sug.control_measures.agricultural_control) {
              formattedText += `【农业防治】\n${sug.control_measures.agricultural_control}\n\n`;
            }

            // 3. 提取化学防治 / 药剂推荐 (chemical_control.recommendations 数组)
            const chem = sug.control_measures?.chemical_control;
            if (chem && chem.recommendations && chem.recommendations.length > 0) {
              formattedText += `【药剂推荐】\n`;
              chem.recommendations.forEach((item, index) => {
                formattedText += `${index + 1}. ${item.agent_name}\n`;
                formattedText += `   用法：${item.usage}\n`;
                formattedText += `   注意：${item.precaution}\n`;
              });
              formattedText += `\n`;
            }

            // 4. 提取预防建议 (prevention_tips)
            if (sug.prevention_tips) {
              formattedText += `【预防建议】\n${sug.prevention_tips}\n\n`;
            }

            // 5. 风险评估 (risk_assessment)
            if (sug.risk_assessment) {
              formattedText += `【风险评估】\n${sug.risk_assessment}`;
            }
          } else {
            formattedText = sug || '暂无详细建议';
          }

          this.setData({
            'currentRecord.suggestion': formattedText,
            'currentRecord.score': detail.health_score || 0,
            isDetailLoading: false
          });
        }
      } catch (e) {
        console.error("详情获取失败", e);
        this.setData({
          isDetailLoading: false
        });
      }
    }
  },

  // 🔥 关闭弹窗
  closeModal() {
    this.setData({
      showModal: false
    })
  },

  // 🔥 禁止点击弹窗内容时关闭
  stopClose() {},

  // 🔥 删除单条记录
  deleteRecord(e) {
    const index = e.currentTarget.dataset.index
    let records = this.data.records

    wx.showModal({
      title: '确认删除',
      content: '确定删除这条记录？',
      success: res => {
        if (res.confirm) {
          records.splice(index, 1)
          this.setData({
            records
          })
          wx.setStorageSync('upload_records', records)
          wx.showToast({
            title: '已删除'
          })
        }
      }
    })
  },
  // 🔥 新增：加载云端记录
  // 加载云端上传记录
  // 加载云端上传记录
  async loadCloudRecords() {
    const userInfo = getApp().globalData.userInfo || wx.getStorageSync('userInfo');
    // 2. 判断是否为游客（如果没有 openid 或者明确没有登录信息）
    if (!userInfo || !userInfo.openid) {
      console.log("当前为游客模式，不加载云端记录");
      this.setData({
        cloudRecords: [],
        isLoading: false
      });
      return; // 🔥 游客直接跳出，不执行云函数请求
    }

    this.setData({
      isLoading: true
    });

    try {
      const res = await wx.cloud.callFunction({
        name: 'getRecord',
        data: {
          page: 1,
          pageSize: 50
        }
      });

      if (res.result && res.result.code === 0) {
        const formattedRecords = res.result.data.records.map(item => {
          // 1. 处理时区与格式化 (统一为：YYYY-MM-DD HH:mm:ss)
          const date = new Date(item.createdAt);

          const Y = date.getFullYear();
          const M = (date.getMonth() + 1).toString().padStart(2, '0');
          const D = date.getDate().toString().padStart(2, '0');
          const h = date.getHours().toString().padStart(2, '0');
          const m = date.getMinutes().toString().padStart(2, '0');
          const s = date.getSeconds().toString().padStart(2, '0');

          const timeStr = `${Y}-${M}-${D} ${h}:${m}:${s}`;

          // 2. 统一字段映射，确保与本地记录完全一致
          return {
            ...item,
            imgUrl: item.image_cloud_id,
            time: timeStr, // 强制统一字段名为 time
            type: item.result[0].diseaseName || '未知',
            level: item.health_score >= 80 ? '轻微' : (item.health_score >= 60 ? '中度' : '严重'),
            status: '已同步'
          };
        });

        this.setData({
          cloudRecords: formattedRecords
        });
      }
    } catch (e) {
      console.error("云端数据请求失败", e);
      wx.showToast({
        title: '同步失败',
        icon: 'none'
      });
    } finally {
      this.setData({
        isLoading: false
      });
    }
  },
  // 修改原有的 switchTab，切换时触发加载
  switchTab(e) {
    const tab = Number(e.currentTarget.dataset.tab);
    this.setData({
      currentTab: tab
    });

    // 如果切换到云端且目前没数据，则自动加载
    if (tab === 1 && this.data.cloudRecords.length === 0) {
      this.loadCloudRecords();
    }
  },
  deleteCloudRecord(e) {
    const {
      id,
      index
    } = e.currentTarget.dataset;
    const that = this;

    wx.showModal({
      title: '确认删除',
      content: '确定从云端永久删除这条记录？',
      confirmColor: '#ff4d4f',
      success: async (res) => {
        if (res.confirm) {
          wx.showLoading({
            title: '正在删除...',
            mask: true
          });

          try {
            // 调用云函数，注意参数名为 recordIds 且为数组格式
            const result = await wx.cloud.callFunction({
              name: 'deleteRecord',
              data: {
                recordIds: [id]
              }
            });

            if (result.result && result.result.code === 0) {
              // 删除成功后更新前端列表，实现“无感删除”
              let cloudRecords = that.data.cloudRecords;
              cloudRecords.splice(index, 1);
              that.setData({
                cloudRecords
              });

              wx.hideLoading();
              wx.showToast({
                title: '已删除',
                icon: 'success'
              });
            } else {
              throw new Error(result.result.msg || '后端返回错误');
            }
          } catch (err) {
            wx.hideLoading();
            console.error('云端删除失败:', err);
            wx.showToast({
              title: '删除失败',
              icon: 'none'
            });
          }
        }
      }
    });
  },
  // 🛠️ 辅助函数1：动态测量文本在指定宽度下，换行后所需的总高度
  measureTextHeight(ctx, text, font, maxWidth, lineHeight) {
    if (!text) return 0;
    ctx.font = font;
    let lines = 0;
    let currentLine = '';

    for (let n = 0; n < text.length; n++) {
      const char = text[n];
      // 如果遇到用户输入的主动换行符
      if (char === '\n') {
        lines++;
        currentLine = '';
        continue;
      }
      const testLine = currentLine + char;
      const testWidth = ctx.measureText(testLine).width;
      if (testWidth > maxWidth && n > 0) {
        lines++;
        currentLine = char;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) lines++;
    return lines * lineHeight;
  },

  // 🛠️ 辅助函数2：支持自动换行的文本绘制函数
  drawTextMultiline(ctx, text, x, y, font, color, maxWidth, lineHeight) {
    if (!text) return;
    ctx.font = font;
    ctx.fillStyle = color;
    let currentLine = '';
    let startY = y;

    for (let n = 0; n < text.length; n++) {
      const char = text[n];
      if (char === '\n') {
        ctx.fillText(currentLine, x, startY);
        startY += lineHeight;
        currentLine = '';
        continue;
      }
      const testLine = currentLine + char;
      const testWidth = ctx.measureText(testLine).width;
      if (testWidth > maxWidth && n > 0) {
        ctx.fillText(currentLine, x, startY);
        startY += lineHeight;
        currentLine = char;
      } else {
        currentLine = testLine;
      }
    }
    if (currentLine) {
      ctx.fillText(currentLine, x, startY);
    }
  },

  // 🎯 主函数：动态计算高度并生成完整长图海报
  async drawAndSaveImage() {
    if (this.data.isDetailLoading) {
      wx.showModal({
        title: '请稍候',
        content: '防治建议正在深度解析与同步中，请等待加载完成后再分享。',
        showCancel: false,
        confirmText: '我知道了'
      });
      return; // 拦截，不执行后面的绘图
    }
    wx.showLoading({
      title: '正在深度解析并生成海报...',
      mask: true
    });
    const record = this.data.currentRecord;

    if (!record) {
      wx.hideLoading();
      return;
    }

    // 提取防治建议文本（兼容对象模式或纯文本模式）
    let suggestionText = '';
    if (typeof record.suggestion === 'object') {
      // 这里的结构顺应你数据库返回的实际字段进行微调
      suggestionText = record.suggestion.control_measures?.agricultural_control || '';
    } else {
      suggestionText = record.suggestion || '';
    }

    if (!suggestionText) suggestionText = '暂无防治建议。';

    try {
      const query = wx.createSelectorQuery();
      query.select('#shareCanvas')
        .fields({
          node: true,
          size: true
        })
        .exec(async (res) => {
          if (!res[0] || !res[0].node) {
            wx.hideLoading();
            wx.showToast({
              title: '画布初始化失败',
              icon: 'none'
            });
            return;
          }

          const canvas = res[0].node;
          const ctx = canvas.getContext('2d');

          // --- 布局常量配置 ---
          const POSTER_WIDTH = 320; // 海报固定宽度 (px)
          const IMAGE_HEIGHT = 200; // 图片固定高度 (px)
          const PADDING = 20; // 左右边距 (px)
          const CONTENT_MAX_WIDTH = POSTER_WIDTH - (PADDING * 2); // 文本最大可用宽度

          const INFO_BLOCK_Y = IMAGE_HEIGHT + 40; // 基本检测信息起始 Y 坐标
          const SUGGEST_TITLE_Y = INFO_BLOCK_Y + 140; // “防治建议”四个大字起始 Y 坐标
          const SUGGEST_CONTENT_Y = SUGGEST_TITLE_Y + 30; // 建议正文内容起始 Y 坐标

          const SUGGEST_FONT = '12px sans-serif'; // 建议正文字体
          const SUGGEST_LINE_HEIGHT = 18; // 建议正文行高

          // 1. 【核心突破】在清空画布前，利用辅助函数预估长文本的高度
          ctx.font = SUGGEST_FONT;
          const textBlockHeight = this.measureTextHeight(ctx, suggestionText, SUGGEST_FONT, CONTENT_MAX_WIDTH, SUGGEST_LINE_HEIGHT);

          // 2. 【动态计算总高】基础高度 + 动态文字高度 + 底部留白保底
          const TOTAL_POSTER_HEIGHT = SUGGEST_CONTENT_Y + textBlockHeight + 30;

          // 3. 重新对画布进行高清缩放赋值（会重置画布状态）
          const dpr = wx.getSystemInfoSync().pixelRatio;
          canvas.width = POSTER_WIDTH * dpr;
          canvas.height = TOTAL_POSTER_HEIGHT * dpr;
          ctx.scale(dpr, dpr);

          // 4. 开始绘制海报背景
          ctx.fillStyle = '#FFFFFF';
          ctx.fillRect(0, 0, POSTER_WIDTH, TOTAL_POSTER_HEIGHT);

          // 5. 异步加载并绘制主图
          const mainImg = canvas.createImage();
          let httpSrc = record.imgUrl;
          if (httpSrc.startsWith('cloud://')) {
            const tempRes = await wx.cloud.getTempFileURL({
              fileList: [httpSrc]
            });
            httpSrc = tempRes.fileList[0].tempFileURL;
          }

          mainImg.src = httpSrc;
          await new Promise((resolve, reject) => {
            mainImg.onload = resolve;
            mainImg.onerror = reject;
          });

          // 📐 定义图片在海报上的坐标和大小
          const imgRectX = 15;
          const imgRectY = 15;
          const imgRectW = POSTER_WIDTH - 30; // 290px
          const imgRectH = IMAGE_HEIGHT; // 200px
          const cornerRadius = 8; // 圆角大小，如果不需要圆角可以设为 0

          // 💡【核心修正】：加入圆角裁剪与等比例居中算法
          ctx.save(); // 保存当前画布状态

          // 如果需要圆角，调用辅助函数创建路径并裁剪
          if (cornerRadius > 0) {
            this.drawRoundedRectPath(ctx, imgRectX, imgRectY, imgRectW, imgRectH, cornerRadius);
            ctx.clip(); // 将画布裁剪成圆角矩形区域
          }

          // 🎯 调用刚才写好的等比例绘制函数，彻底解决图片扭曲问题！
          this.drawImagePreserveRatio(ctx, mainImg, imgRectX, imgRectY, imgRectW, imgRectH);

          ctx.restore();

          // 6. 绘制中间的检测数据项
          const isHealthy = record.score === 100;
          ctx.fillStyle = isHealthy ? '#07c160' : '#ff4444';
          ctx.font = 'bold 16px sans-serif';
          ctx.fillText(`检测结果：${isHealthy ? '未发现病害' : '存在病害'}`, 20, INFO_BLOCK_Y);

          ctx.fillStyle = '#333333';
          ctx.font = '13px sans-serif';
          ctx.fillText(`图像增强：${record.result[0].diseaseName || '无'}`, 20, INFO_BLOCK_Y + 30);
          ctx.fillText(`启智模式：${record.result[0].level || '无'}`, 20, INFO_BLOCK_Y + 60);
          ctx.fillText(`检测地点：${record.location.city || '未知'}`, 20, INFO_BLOCK_Y + 90);
          ctx.fillText(`记录时间：${record.time || '未知'}`, 20, INFO_BLOCK_Y + 120);

          // 7. 绘制“防治建议”小标题
          ctx.fillStyle = '#000000';
          ctx.font = 'bold 15px sans-serif';
          ctx.fillText('防治建议：', 20, SUGGEST_TITLE_Y);

          // 8. 【核心修正】调用多行绘制函数，完美渲染完整长文本
          this.drawTextMultiline(
            ctx,
            suggestionText,
            20,
            SUGGEST_CONTENT_Y,
            SUGGEST_FONT,
            '#666666',
            CONTENT_MAX_WIDTH,
            SUGGEST_LINE_HEIGHT
          );

          // 9. 将画布导出并唤起相册保存
          wx.canvasToTempFilePath({
            canvas: canvas,
            success: (exportRes) => {
              this.saveToAlbum(exportRes.tempFilePath);
            },
            fail: (err) => {
              wx.hideLoading();
              wx.showToast({
                title: '生成图片失败',
                icon: 'none'
              });
              console.error(err);
            }
          });
        });
    } catch (e) {
      wx.hideLoading();
      console.error(e);
      wx.showToast({
        title: '海报生成出错',
        icon: 'none'
      });
    }
  },

  // 辅助保存方法（保持不变）
  saveToAlbum(filePath) {
    wx.saveImageToPhotosAlbum({
      filePath: filePath,
      success: () => {
        wx.hideLoading();
        wx.showModal({
          title: '保存成功',
          content: '包含完整防治建议的海报已保存到您的手机相册',
          showCancel: false
        });
      },
      fail: (err) => {
        wx.hideLoading();
        if (err.errMsg.includes('auth deny')) {
          wx.showModal({
            title: '授权提示',
            content: '保存图片需要您开启相册写入权限',
            success: (res) => {
              if (res.confirm) wx.openSetting();
            }
          });
        }
      }
    });
  },
  /**
   * 🛠️ 辅助函数：等比例居中裁剪绘制图片（类似 CSS 的 object-fit: cover）
   * sw, sh 是原图截取的宽高；sx, sy 是截取的起始坐标
   */
  drawImagePreserveRatio(ctx, imgNode, x, y, width, height) {
    if (!imgNode.width || !imgNode.height) return;

    const canvasWidth = width;
    const canvasHeight = height;
    const imageWidth = imgNode.width;
    const imageHeight = imgNode.height;

    // 1. 计算缩放比例（取宽高缩放比中较大的一个，确保图片能铺满整个框）
    const scale = Math.max(canvasWidth / imageWidth, canvasHeight / imageHeight);

    // 2. 根据缩放比，反推出在原图上需要截取的区域大小
    const sw = canvasWidth / scale;
    const sh = canvasHeight / scale;

    // 3. 计算截取起始点，使其居中
    const sx = (imageWidth - sw) / 2;
    const sy = (imageHeight - sh) / 2;

    // 4. 使用 9 参数版本的 drawImage 进行精准截取和绘制
    ctx.drawImage(imgNode, sx, sy, sw, sh, x, y, canvasWidth, canvasHeight);
  },

  /**
   * 🛠️ 辅助函数：绘制圆角矩形路径（用于裁剪圆角图片，让界面更好看）
   */
  drawRoundedRectPath(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }
})