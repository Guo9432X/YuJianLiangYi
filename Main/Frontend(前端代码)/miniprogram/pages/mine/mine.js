Page({
  data: {
    showHelpModal: false,
    userInfo: null,
    cacheSize: '0KB'
  },
  // 🎯 核心：点击“使用帮助”的点击事件
  ShowHelp() {
    this.setData({
      showHelpModal: true // 打开弹窗
    });
  },

  // 关闭帮助弹窗
  closeHelpModal() {
    this.setData({
      showHelpModal: false // 关闭弹窗
    });
  },
  
  stopClose() {
    // 空函数，用来阻止冒泡，防止点击弹窗内部内容时误关闭弹窗
  },

  onShow() {
    this.getUserInfo();
    this.calculateCache();
  },

  // 1. 获取登录信息
  getUserInfo() {
    const info = getApp().globalData.userInfo || wx.getStorageSync('userInfo');
    this.setData({ userInfo: info });
  },

  // 2. 计算当前缓存大小 (主要计算本地 upload_records)
  calculateCache() {
    try {
      // 1. 定向获取你想计算的那个缓存
      const records = wx.getStorageSync('upload_records');
      
      // 2. 如果数据不存在（比如刚被删了），直接硬编码为 0
      if (!records) {
        this.setData({ cacheSize: '0 KB' });
        return;
      }
      
      // 3. 如果有数据，计算这个数据的实际体积（大致估算：字符数 * 2 字节）
      const str = JSON.stringify(records);
      const sizeInBytes = str.length * 2; 
      const sizeInKB = sizeInBytes / 1024;
      
      // 4. 动态转化为 KB 或 MB 显示
      this.setData({
        cacheSize: sizeInKB > 1024 
          ? (sizeInKB / 1024).toFixed(2) + ' MB' 
          : sizeInKB.toFixed(2) + ' KB'
      });
      
    } catch (e) {
      console.error('计算缓存失败:', e);
      // 降级处理，避免界面卡住
      this.setData({ cacheSize: '0 KB' });
    }
  },

  // 3. 执行清除缓存
  handleClearCache() {
    wx.showModal({
      title: '提示',
      content: '确定要清除所有本地上传记录吗？（云端同步记录不受影响）',
      confirmColor: '#ff4d4f',
      success: (res) => {
        if (res.confirm) {
          try {
            // 我们只清除特定的记录 key，不建议使用 clearStorageSync() 以免误删登录态
            wx.removeStorageSync('upload_records');
            wx.showToast({ title: '清理完成', icon: 'success' });
            this.calculateCache(); // 刷新显示
          } catch (e) {
            wx.showToast({ title: '清理失败', icon: 'none' });
          }
        }
      }
    });
  },

  // 4. 登录跳转 (根据你的实际登录页面路径修改)
  toLogin() {
    wx.navigateTo({ url: '/pages/index/index' });
  },

  // 5. 退出登录
// 5. 退出登录
handleLogout() {
  wx.showModal({
    title: '提示',
    content: '确定要退出登录吗？',
    success: (res) => {
      if (res.confirm) {
        // 1. 清除登录态
        getApp().globalData.userInfo = null;
        wx.removeStorageSync('userInfo');
        
        // 2. 更新当前页面显示
        this.setData({ userInfo: null });

        // 3. 提示并跳转
        wx.showToast({ 
          title: '已退出',
          icon: 'success',
          duration: 1000,
          success: () => {
            // 延迟一秒跳转，让用户看清“已退出”提示
            setTimeout(() => {
              // ✅ 如果目标页是 TabBar 页面（如下面的 index），必须用 switchTab
              /*wx.switchTab({
                url: '/pages/index/index' 
              });*/
               // ❌ 如果目标页不是底部菜单栏页面，则用 redirectTo
              wx.redirectTo({
                url: '/pages/index/index'
              });
              
            }, 1000);
          }
        });
      }
    }
  });
},
// pages/me/me.js

handleFeedback() {
  wx.showModal({
    title: '意见反馈',
    placeholderText: '请输入您的建议或遇到的问题...',
    editable: true, // 开启原生输入框
    success: (res) => {
      // 用户点击了确定
      if (res.confirm) {
        const feedbackContent = res.content ? res.content.trim() : '';

        // 1. 前端简单校验
        if (!feedbackContent) {
          wx.showToast({ title: '内容不能为空', icon: 'none' });
          return;
        }

        wx.showLoading({ title: '提交中...', mask: true });

        // 2. 调用云函数 submitFeedback
        wx.cloud.callFunction({
          name: 'submitFeedback', // 云函数名称
          data: {
            content: feedbackContent // 对应云函数中的 event.content
          },
          success: (cloudRes) => {
            wx.hideLoading();
            // 检查云函数返回的 code
            if (cloudRes.result && cloudRes.result.code === 0) {
              wx.showToast({
                title: '提交成功',
                icon: 'success'
              });
            } else {
              wx.showModal({
                title: '提交失败',
                content: cloudRes.result.msg || '未知错误',
                showCancel: false
              });
            }
          },
          fail: (err) => {
            wx.hideLoading();
            console.error('[云函数调用失败]', err);
            wx.showModal({
              title: '网络异常',
              content: '反馈提交失败，请稍后再试',
              showCancel: false
            });
          }
        });
      }
    }
  });
}
});