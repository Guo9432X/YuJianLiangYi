// app.js
App({
  onLaunch() {
    // 1. 获取唯一场景值，作为游客缓存的区分键
    const options = wx.getLaunchOptionsSync()
    this.globalData.scene = options.scene || 'default_123'

    // 2. 云开发环境初始化
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上基础库以使用云能力');
    } else {
      wx.cloud.init({
        // env 参数说明：使用你提供的环境 ID
        //env: 'cloud1-9gp9jcwedb8e068b',
        env: 'corn-diseases-d5gjfzoat0f031328',
        // traceUser: true 会在云开发控制台记录用户访问记录
        traceUser: true,
      });
    }

    // 3. (可选) 启动时检查本地是否已有登录缓存
    const userInfo = wx.getStorageSync('userInfo');
    if (userInfo) {
      this.globalData.userInfo = userInfo;
    }
  },

  globalData: {
    scene: '',
    userInfo: null
  }
})